"""
유방암 RAG 프로젝트 - 데이터 수집 스크립트
================================================

P0 ~ P1 수준의 공식 의료 자료를 자동 다운로드합니다.

실행 방법:
    pip install requests beautifulsoup4 tqdm
    python breast_rag_download.py

저장 구조:
    data/raw/pdf/<출처>/  -> PDF 파일
    data/raw/html/<출처>/ -> 웹페이지 HTML
    data/raw/metadata/manifest.json -> 모든 파일 메타데이터
    data/raw/download_log.txt -> 다운로드 로그
"""

import json
import time
from pathlib import Path
from datetime import datetime
from urllib.parse import urlparse

import requests
from tqdm import tqdm


# ============================================================
# 설정
# ============================================================

BASE_DIR = Path("data/raw")
PDF_DIR = BASE_DIR / "pdf"
HTML_DIR = BASE_DIR / "html"
META_DIR = BASE_DIR / "metadata"
LOG_FILE = BASE_DIR / "download_log.txt"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "application/pdf,text/html,application/xhtml+xml,*/*;q=0.8",
    "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7",
}

TIMEOUT = 60  # seconds
SLEEP_BETWEEN = 1.5  # 사이트 부담 줄이기 위한 딜레이 (초)


# ============================================================
# 수집 대상 목록
# ============================================================

# PDF 자료 (출처별로 폴더 분리)
PDF_SOURCES = [
    # ---------- P0: 국내 핵심 가이드라인 ----------
    {
        "priority": "P0",
        "source": "kbcs",
        "org": "대한유방암학회",
        "title": "제10차 한국유방암 진료권고안 (2023)",
        "filename": "kbcs_korean_breast_cancer_guideline_2023.pdf",
        "url": "https://ilsan.m.chamc.co.kr/asset/file/%EC%9C%A0%EB%B0%A9%EC%95%94_%EC%A7%84%EB%A3%8C%EA%B6%8C%EA%B3%A0%EC%95%88(2023).pdf",
        "language": "ko",
        "doc_type": "clinical_guideline",
    },
    # ---------- P0: 국립암센터 / 국가암정보센터 ----------
    {
        "priority": "P0",
        "source": "ncc",
        "org": "국립암센터",
        "title": "유방암 검진 권고안 (2015)",
        "filename": "ncc_breast_cancer_screening_guideline_2015.pdf",
        "url": "https://www.cancer.go.kr/download.do?uuid=d6c68b55-5c38-455e-b496-669dce2f2127.pdf",
        "language": "ko",
        "doc_type": "screening_guideline",
    },
    {
        "priority": "P0",
        "source": "ncc",
        "org": "국립암센터",
        "title": "국민 암예방 수칙 실천지침 - 유방암",
        "filename": "ncc_breast_cancer_prevention_guide.pdf",
        "url": "https://www.cancer.go.kr/download.do?uuid=934546bb-2ba9-4ff4-a88a-664580e126b6.pdf",
        "language": "ko",
        "doc_type": "patient_guide",
    },
    {
        "priority": "P0",
        "source": "ncc",
        "org": "국립암센터",
        "title": "유방암 항호르몬제 안내",
        "filename": "ncc_breast_cancer_hormone_therapy.pdf",
        "url": "https://www.cancer.go.kr/download.do?uuid=08e91cf1-3933-465d-8021-8df7af09c9eb.pdf",
        "language": "ko",
        "doc_type": "patient_guide",
    },
    # ---------- P1: NCCN 환자용 가이드라인 (영어) ----------
    {
        "priority": "P1",
        "source": "nccn",
        "org": "NCCN",
        "title": "NCCN Guidelines for Patients: Invasive Breast Cancer",
        "filename": "nccn_invasive_breast_cancer_patient.pdf",
        "url": "https://www.nccn.org/patients/guidelines/content/PDF/breast-invasive-patient.pdf",
        "language": "en",
        "doc_type": "patient_guideline",
    },
    {
        "priority": "P1",
        "source": "nccn",
        "org": "NCCN",
        "title": "NCCN Guidelines for Patients: Ductal Carcinoma In Situ (DCIS)",
        "filename": "nccn_dcis_patient.pdf",
        "url": "https://www.nccn.org/patients/guidelines/content/PDF/stage_0_breast-patient.pdf",
        "language": "en",
        "doc_type": "patient_guideline",
    },
    {
        "priority": "P1",
        "source": "nccn",
        "org": "NCCN",
        "title": "NCCN Guidelines for Patients: Metastatic Breast Cancer",
        "filename": "nccn_metastatic_breast_cancer_patient.pdf",
        "url": "https://www.nccn.org/patients/guidelines/content/PDF/stage_iv_breast-patient.pdf",
        "language": "en",
        "doc_type": "patient_guideline",
    },
    {
        "priority": "P1",
        "source": "nccn",
        "org": "NCCN",
        "title": "NCCN Guidelines for Patients: Breast Cancer Screening and Diagnosis",
        "filename": "nccn_breast_cancer_screening_diagnosis_patient.pdf",
        "url": "https://www.nccn.org/patients/guidelines/content/PDF/breastcancerscreening-patient.pdf",
        "language": "en",
        "doc_type": "patient_guideline",
    },
    {
        "priority": "P1",
        "source": "nccn",
        "org": "NCCN",
        "title": "NCCN Guidelines for Patients: Inflammatory Breast Cancer",
        "filename": "nccn_inflammatory_breast_cancer_patient.pdf",
        "url": "https://www.nccn.org/patients/guidelines/content/PDF/inflammatory-breast-patient.pdf",
        "language": "en",
        "doc_type": "patient_guideline",
    },
    # ---------- P1: ESMO 한국어 환자 안내서 ----------
    {
        "priority": "P1",
        "source": "esmo",
        "org": "ESMO",
        "title": "ESMO Breast Cancer Guide for Patients (Korean)",
        "filename": "esmo_breast_cancer_patient_guide_korean.pdf",
        "url": "https://dam.esmo.org/image/upload/v1742457711/KO_%7C_Breast_Cancer%3A_Guide_for_Patients_%E2%80%93_Korean.pdf",
        "language": "ko",
        "doc_type": "patient_guideline",
    },
]

# 웹 페이지 자료 (보조 corpus용)
HTML_SOURCES = [
    {
        "priority": "P0",
        "source": "ncic",
        "org": "국가암정보센터",
        "title": "내가 알고 싶은 암 - 유방암",
        "filename": "ncic_breast_cancer_main.html",
        "url": "https://www.cancer.go.kr/lay1/program/S1T211C217/cancer/view.do?cancer_seq=4757",
        "language": "ko",
        "doc_type": "patient_info_web",
    },
    {
        "priority": "P1",
        "source": "snuh",
        "org": "서울대학교암병원",
        "title": "유방센터 안내",
        "filename": "snuh_breast_center.html",
        "url": "https://cancer.snuh.org/reservation/meddept/BCCG/cancerIntro.do",
        "language": "ko",
        "doc_type": "hospital_info_web",
    },
    {
        "priority": "P1",
        "source": "smc",
        "org": "삼성서울병원",
        "title": "유방암 정보",
        "filename": "smc_breast_cancer.html",
        "url": "https://www.samsunghospital.com/dept/main/index.do?DP_CODE=1812J2&MENU_ID=004031028",
        "language": "ko",
        "doc_type": "hospital_info_web",
    },
]


# ============================================================
# 유틸 함수
# ============================================================

def setup_dirs() -> None:
    """필요한 디렉토리 생성."""
    BASE_DIR.mkdir(parents=True, exist_ok=True)
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    HTML_DIR.mkdir(parents=True, exist_ok=True)
    META_DIR.mkdir(parents=True, exist_ok=True)
    for item in PDF_SOURCES:
        (PDF_DIR / item["source"]).mkdir(parents=True, exist_ok=True)
    for item in HTML_SOURCES:
        (HTML_DIR / item["source"]).mkdir(parents=True, exist_ok=True)


def log(msg: str) -> None:
    """로그 파일과 콘솔에 동시 기록."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def is_valid_pdf(filepath: Path) -> bool:
    """PDF 매직 넘버 확인 (%PDF-)."""
    try:
        with open(filepath, "rb") as f:
            head = f.read(5)
        return head == b"%PDF-"
    except Exception:
        return False


def download_file(url: str, target: Path, is_pdf: bool = True) -> tuple[bool, str]:
    """파일 다운로드. 성공 여부와 상태 메시지 반환."""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=TIMEOUT, stream=True)
        resp.raise_for_status()

        with open(target, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

        size = target.stat().st_size
        if size < 1024:
            return False, f"파일이 너무 작음 ({size} bytes)"

        if is_pdf and not is_valid_pdf(target):
            return False, "PDF 시그니처 없음 (HTML 에러 페이지일 가능성)"

        return True, f"OK ({size:,} bytes)"

    except requests.exceptions.HTTPError as e:
        return False, f"HTTP 오류: {e}"
    except requests.exceptions.Timeout:
        return False, "타임아웃"
    except Exception as e:
        return False, f"예외: {type(e).__name__}: {e}"


# ============================================================
# 메인 다운로드 루틴
# ============================================================

def run() -> None:
    setup_dirs()
    log("=" * 60)
    log("유방암 RAG 데이터 수집 시작")
    log("=" * 60)

    manifest = []
    success_count = 0
    fail_count = 0

    # PDF 다운로드
    log(f"\n[PDF] 총 {len(PDF_SOURCES)}개 다운로드 시작")
    for item in tqdm(PDF_SOURCES, desc="PDF"):
        target = PDF_DIR / item["source"] / item["filename"]

        if target.exists() and is_valid_pdf(target):
            log(f"  SKIP (이미 존재): {item['filename']}")
            ok, status = True, "이미 존재"
        else:
            ok, status = download_file(item["url"], target, is_pdf=True)
            if ok:
                log(f"  OK   {item['filename']} -> {status}")
                success_count += 1
            else:
                log(f"  FAIL {item['filename']} -> {status}")
                fail_count += 1
                # 실패한 파일은 삭제 (잘못된 내용이 남지 않도록)
                if target.exists():
                    target.unlink()

            time.sleep(SLEEP_BETWEEN)

        manifest.append({
            **item,
            "local_path": str(target.relative_to(BASE_DIR.parent.parent))
                          if target.exists() else None,
            "downloaded": ok,
            "status": status,
            "downloaded_at": datetime.now().isoformat() if ok else None,
        })

    # HTML 다운로드
    log(f"\n[HTML] 총 {len(HTML_SOURCES)}개 다운로드 시작")
    for item in tqdm(HTML_SOURCES, desc="HTML"):
        target = HTML_DIR / item["source"] / item["filename"]

        if target.exists() and target.stat().st_size > 1024:
            log(f"  SKIP (이미 존재): {item['filename']}")
            ok, status = True, "이미 존재"
        else:
            ok, status = download_file(item["url"], target, is_pdf=False)
            if ok:
                log(f"  OK   {item['filename']} -> {status}")
                success_count += 1
            else:
                log(f"  FAIL {item['filename']} -> {status}")
                fail_count += 1
                if target.exists():
                    target.unlink()

            time.sleep(SLEEP_BETWEEN)

        manifest.append({
            **item,
            "local_path": str(target.relative_to(BASE_DIR.parent.parent))
                          if target.exists() else None,
            "downloaded": ok,
            "status": status,
            "downloaded_at": datetime.now().isoformat() if ok else None,
        })

    # 매니페스트 저장
    manifest_path = META_DIR / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    # 요약
    log("\n" + "=" * 60)
    log(f"완료: 성공 {success_count}건 / 실패 {fail_count}건")
    log(f"매니페스트: {manifest_path}")
    log("=" * 60)

    # 실패 목록 다시 출력
    failed = [m for m in manifest if not m["downloaded"]]
    if failed:
        log("\n[실패 목록 - 수동 다운로드 필요]")
        for m in failed:
            log(f"  - [{m['priority']}] {m['org']} / {m['title']}")
            log(f"    URL: {m['url']}")
            log(f"    이유: {m['status']}")


if __name__ == "__main__":
    run()
