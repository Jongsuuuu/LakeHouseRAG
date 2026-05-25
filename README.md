# 🏛️ LakehouseRAG

> **다중 포맷 비정형 데이터를 위한 지능형 검색 시스템**  
> BM25 + Fuzzy + Semantic 3중 하이브리드 RAG 파이프라인

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/LangChain-latest-green?logo=chainlink)](https://langchain.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red?logo=streamlit)](https://streamlit.io/)
[![Gemini](https://img.shields.io/badge/Gemini-2.5%20Flash-yellow?logo=google)](https://deepmind.google/technologies/gemini/)
[![FAISS](https://img.shields.io/badge/FAISS-CPU-lightgrey?logo=meta)](https://faiss.ai/)

---

## 📌 프로젝트 개요

**LakehouseRAG**는 Google Drive에 저장된 비정형 데이터(TXT, PDF, DOCX, XLSX, CSV, JSON, 이미지 등)를 자동으로 수집·적재하고, **3중 하이브리드 검색 엔진**과 **LLM(Gemini 2.5 Flash)** 을 결합하여 자연어로 데이터를 탐색할 수 있는 지능형 검색 시스템입니다.

단순한 키워드 매칭이나 벡터 검색만으로는 놓치기 쉬운 다양한 쿼리 패턴(정규식 / 단어 / 자연어 문장)을 **AI가 자동으로 분류**하고 최적의 검색 전략을 라우팅합니다. 검색 결과는 Chain-of-Thought 프롬프트가 적용된 Gemini가 다중 문서를 교차 분석하여 출처까지 명시한 구조화된 답변으로 반환합니다.

### 핵심 특징

| 기능 | 내용 |
|------|------|
| 📥 **데이터 적재** | TXT / PDF / DOCX / XLSX / CSV / JSON / 이미지(OCR) + MD5 해시 기반 증분 적재 |
| 🔍 **검색 엔진** | BM25 + Fuzzy(Regex 포함) + Semantic 3중 하이브리드 |
| 🧠 **임베딩** | `paraphrase-multilingual-MiniLM-L12-v2` (한국어·영어 다국어 지원) |
| 🤖 **LLM** | Gemini 2.5 Flash + 슬라이딩 윈도우 대화 메모리 (최대 6턴 유지) |
| 💬 **인터랙티브 UI** | Streamlit 채팅 + 파일 탐색기 + 현황 대시보드 |
| 🔄 **증분 동기화** | MD5 해시 비교 → 변경 파일만 재처리, 불변 파일 스킵 |

---

## 🏗️ 시스템 아키텍처

<img width="1195" height="1545" alt="image" src="https://github.com/user-attachments/assets/edc408f2-95da-4841-90f4-cf1432a7bd4d" />

※ 아키텍처 다이어그램 시각화에는 ChatGPT(OpenAI) 기반 이미지 생성 기능을 활용했습니다.

---

## ⚙️ 핵심 구현 및 최적화

### 1. AI 기반 쿼리 라우팅 (Query Classification)

검색 입력을 **3가지 유형**으로 분류하여 최적의 검색 엔진을 자동 선택합니다.

```python
def classify_query(query: str, llm=None) -> str:
    # LLM이 'regex' | 'word' | 'sentence' 중 하나 반환
    # LLM 실패 시 규칙 기반 fallback 동작
```

| 쿼리 유형 | 예시 | 적용 검색 |
|-----------|------|-----------|
| `regex` | `\d{3}-\d{4}`, `[A-Z]+\d+` | Fuzzy(정규식 분기) |
| `word` | `매출`, `김철수`, `2023년` | BM25 + Fuzzy |
| `sentence` | `지난 분기 매출 현황을 알려줘` | Semantic(FAISS MMR) |

LLM 호출 실패를 대비한 **규칙 기반 fallback** 로직이 구현되어 있어 API 오류가 발생해도 시스템이 정상 동작합니다.

---

### 2. MD5 해시 기반 증분 적재 (Incremental Ingestion)

```python
def sync_lakehouse(data_dir, db_path, hash_log_path):
    # MD5 비교 → 변경 파일만 재처리
    # 기존 DB는 parquet으로 유지 (빠른 I/O)
```

- 파일 변경 여부를 MD5 해시로 판단 → **불필요한 재처리 완전 차단**
- `file_hash_log.json`에 이전 해시 저장 → 재실행 시 즉시 비교
- Parquet 포맷 사용 → 컬럼 기반 압축으로 대용량 텍스트 DB 효율 저장

---

### 3. 다중 포맷 텍스트 추출

| 포맷 | 라이브러리 | 특이사항 |
|------|-----------|----------|
| PDF | PyMuPDF(본문) + pdfplumber(표) | 표를 마크다운 형식으로 별도 추출 |
| DOCX | python-docx | 단락 + 표 셀 모두 수집 |
| XLSX/XLS | pandas | 전체 시트명 포함 문자열화 |
| CSV | pandas | UTF-8 errors='replace' 안전 처리 |
| JSON | json | ensure_ascii=False 한글 보존 |
| 이미지 | EasyOCR (ko+en) | 지연 초기화(Lazy Init)로 불필요한 모델 로딩 방지 |

---

### 4. BM25 + Fuzzy 하이브리드 검색

**BM25 (Okapi BM25)**
- 문서 전체를 토크나이즈하여 역색인 구성
- `top_k=10` 결과 반환 후 FAISS 결과와 병합

**Fuzzy 검색 (한국어 특화)**
- 한글 유니코드 NFD 분해(자모 단위 비교)로 오타 내성 확보
- 첫 글자 불일치 시 유사도 페널티(×0.7) 적용 → 노이즈 억제
- Regex 모드 자동 분기 → 정규식 패턴 검색 지원

```python
# 한글 자모 분해 비교
query_proc = decompose_text(query_clean)  # NFD normalize
score = fuzz.ratio(query_proc, wp)
if query_clean[0] != word[0]:
    score *= 0.7  # 첫 글자 다르면 패널티
```

---

### 5. FAISS MMR (Maximal Marginal Relevance)

```python
retriever = vectorstore.as_retriever(
    search_type='mmr',
    search_kwargs={'k': 20, 'fetch_k': 40}
)
```

단순 유사도 순위 대신 **다양성을 보장하는 MMR**을 채택합니다. `fetch_k=40`개 후보 중 중복을 제거하며 `k=20`개를 선택하여 같은 파일의 유사한 청크가 반복되는 문제를 방지합니다.

---

### 6. Chain-of-Thought 시스템 프롬프트

```
1. [문서 파악] 제공된 문서 목록과 각 파일의 주제를 파악
2. [근거 추출] 질문과 관련된 핵심 내용을 정확히 찾아냄
3. [종합 분석] 여러 문서 정보를 교차 검증하고 종합
4. [답변 구성] 명확하고 구조적인 답변 작성
5. [출처 명시] 📄 출처: [파일명] 형식으로 반드시 표기
```

LLM이 문서에 없는 내용을 추측하지 않도록 강제하며, 검색 라우팅 힌트를 `{routing_hint}` 변수로 주입하여 검색 결과를 적절히 반영합니다.

---

### 7. 슬라이딩 윈도우 대화 메모리

```python
@dataclass
class ConversationMemory:
    max_turns: int = 6   # 최근 6턴(12 메시지)만 유지
    history: deque = field(default_factory=deque)
```

`collections.deque`를 사용해 오래된 대화는 자동으로 제거하여 **LLM 컨텍스트 윈도우 초과를 방지**하면서 연속 대화 맥락을 유지합니다.

---

### 8. Streamlit 성능 최적화

```python
@st.cache_resource  # 임베딩 모델, FAISS 인덱스 → 앱 수명 동안 1회만 로드
@st.cache_data      # 레이크하우스 DataFrame → trigger 변경 시에만 재로드
```

`_trigger` 파라미터를 통해 수동 새로고침 시에만 캐시를 무효화하는 전략을 사용합니다. 무거운 임베딩 모델과 FAISS 인덱스를 세션 내에서 재사용하여 응답 지연을 최소화합니다.

---

## 🛠️ 기술 스택

### Core  
![Gemini](https://img.shields.io/badge/Gemini%202.5%20Flash-4285F4?style=flat-square&logo=google&logoColor=white)
![MiniLM](https://img.shields.io/badge/MiniLM%20Multilingual-FFD21E?style=flat-square&logo=huggingface&logoColor=black)
![FAISS](https://img.shields.io/badge/FAISS%20CPU-0467DF?style=flat-square&logo=meta&logoColor=white)
![BM25](https://img.shields.io/badge/rank__bm25-B45309?style=flat-square&logoColor=white)
![thefuzz](https://img.shields.io/badge/thefuzz-1a1a1a?style=flat-square&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?style=flat-square&logo=langchain&logoColor=white)

### 데이터 처리
![PyMuPDF](https://img.shields.io/badge/PyMuPDF-B91C1C?style=flat-square&logoColor=white)
![pdfplumber](https://img.shields.io/badge/pdfplumber-2b6cb0?style=flat-square&logoColor=white)
![python-docx](https://img.shields.io/badge/python--docx-3730A3?style=flat-square&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-150458?style=flat-square&logo=pandas&logoColor=white)
![openpyxl](https://img.shields.io/badge/openpyxl-15803D?style=flat-square&logoColor=white)
![EasyOCR](https://img.shields.io/badge/EasyOCR-C05621?style=flat-square&logoColor=white)
![Parquet](https://img.shields.io/badge/Apache%20Parquet-374151?style=flat-square&logo=apacheparquet&logoColor=white)

### UI & 인프라  
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)
![ipywidgets](https://img.shields.io/badge/ipywidgets-374151?style=flat-square&logo=jupyter&logoColor=white)
![Cloudflare](https://img.shields.io/badge/Cloudflare%20Tunnel-F38020?style=flat-square&logo=cloudflare&logoColor=white)
![Colab](https://img.shields.io/badge/Google%20Colab-F9AB00?style=flat-square&logo=googlecolab&logoColor=black)
![Drive](https://img.shields.io/badge/Google%20Drive-0F9D58?style=flat-square&logo=googledrive&logoColor=white)

---

## 🚀 설치 및 실행 방법

### 1. 사전 준비

**Google Drive 폴더 구조 생성:**
```
/content/drive/MyDrive/
└── DATA_LAKEHOUSE/
    ├── DATA/          ← 여기에 분석할 파일들을 넣어주세요
    │   ├── 보고서.pdf
    │   ├── 데이터.xlsx
    │   ├── 문서.docx
    │   └── ...
    └── api_key.txt    ← Google API Key 파일
```

**`api_key.txt` 형식:**
```
GOOGLE_API_KEY = AIzaSy...your_key_here...
```

> Google AI Studio (https://aistudio.google.com) 에서 무료 API 키를 발급받을 수 있습니다.

---

### 2. 노트북 실행

Google Colab에서 `LakehouseRAG.ipynb`를 열고 순서대로 셀을 실행합니다.

**① 패키지 설치**
```python
!pip install -q \
    langchain langchain-community langchain-google-genai langchain-huggingface \
    faiss-cpu sentence-transformers \
    rank_bm25 thefuzz python-Levenshtein \
    pymupdf pdfplumber python-docx openpyxl \
    easyocr pillow \
    pandas pyarrow \
    streamlit pyngrok \
    google-generativeai

!pip install -q langchain-classic
```

**② Google Drive 마운트 & API 키 로드**
```python
from google.colab import drive
drive.mount('/content/drive')
# → 이후 셀에서 자동으로 API_KEY_PATH에서 키를 파싱
```

**③~⑥ 모듈 정의 및 엔진 초기화**
- `sync_lakehouse()` 실행 → DATA 폴더 자동 스캔 및 Parquet 저장
- `LakehouseEngine` 초기화 → FAISS 인덱스 빌드

**⑦ Colab 인터랙티브 UI로 바로 검색:**
```
# 셀 실행 시 위젯 UI 표시
# 검색어 입력 → 엔터 또는 [검색] 버튼
# 'q' 또는 'exit' 입력 시 종료
```

---

### 3. Streamlit 앱 실행 (선택)

```bash
# 기존 프로세스 종료
!pkill -f streamlit
!pkill -f cloudflared

# Streamlit 앱 백그라운드 실행
!nohup streamlit run app.py \
  --server.port 8501 \
  --server.headless true \
  --server.enableCORS false \
  --server.enableXsrfProtection false \
  > streamlit.log 2>&1 &

# Cloudflare 터널로 외부 URL 생성
!wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -O cloudflared
!chmod +x cloudflared
!nohup ./cloudflared tunnel --url http://localhost:8501 > tunnel.log 2>&1 &

# 접속 URL 확인 (약 5~10초 후)
!grep -Eo "https://[^ ]+trycloudflare\.com" tunnel.log | head -n 1
```

브라우저에서 출력된 `trycloudflare.com` URL로 접속하면 Streamlit UI가 표시됩니다.

---

## 📁 폴더 구조

```
## 📁 폴더 구조
[Google Drive] /DATA_LAKEHOUSE/
├── DATA/                            # 원천 비정형 데이터 폴더
│   ├── *.txt / *.pdf / *.docx
│   ├── *.xlsx / *.xls / *.csv
│   ├── *.json
│   └── *.png / *.jpg / ...
├── LakehouseRAG.ipynb               # 메인 노트북 (전체 파이프라인)
├── api_key.txt                      # 형식: GOOGLE_API_KEY = AIzaSy...
├── file_hash_log.json               # MD5 해시 로그 (증분 적재 기준)
└── structured_lakehouse.parquet     # 추출된 텍스트 구조화 DB
[Colab 런타임 /content/]             # 노트북 실행 후 자동 생성 (임시)
├── app.py                           # ⑧번 셀(%%writefile) 실행 시 생성
└── faiss_index_db/                  # FAISS 인덱스 (런타임 재시작 시 재빌드)
```

* 프로젝트에 사용된 비정형 데이터는 뉴스 기사 외에 모두 가상의 문서입니다.

---


## 🔥 주요 트러블슈팅

### 1. 한국어 Fuzzy 검색 정확도 문제

**문제:** 영어 기반의 Levenshtein 거리 알고리즘은 한글 복합 자모 구조를 단순 문자 단위로 처리하여 실제 발음/의미 유사도를 반영하지 못했습니다. 예를 들어 `김` vs `긲`은 자모 분해 시 거의 동일하지만, 완성형 비교에서는 완전히 다른 문자로 판단됩니다.

**해결:** `unicodedata.normalize('NFD', text)`를 통해 한글을 초성·중성·종성으로 분해한 뒤 비교하도록 전처리 계층을 추가했습니다. 추가로 첫 글자(초성)가 다를 경우 유사도에 ×0.7 페널티를 부여하여 한국어에서 오탐(False Positive)을 억제했습니다.

```python
query_proc = decompose_text(query_clean)  # NFD 자모 분해
if query_clean[0] != word[0]:
    score *= 0.7  # 첫 글자 다를 경우 페널티
```

---

### 2. PDF 표 데이터 유실

**문제:** PyMuPDF(`fitz`)의 `get_text()`는 PDF의 텍스트 레이어만 추출하기 때문에, 표(table) 형식의 데이터를 줄글로 뭉개거나 셀 구분자 없이 나열하는 문제가 있었습니다. 표가 많은 보고서나 재무제표에서 데이터 손실이 발생했습니다.

**해결:** PyMuPDF로 본문 텍스트를 추출하고, `pdfplumber`로 표를 별도 추출하여 **마크다운 테이블 형식**으로 변환 후 본문 뒤에 `[PDF 표 데이터]` 섹션으로 추가하는 이중 추출 전략을 도입했습니다.

```python
body = fitz.open(path)  # 본문 텍스트
tables = extract_pdf_tables(path)  # pdfplumber → 마크다운 표
text = body + "\n\n[PDF 표 데이터]\n" + tables
```

---

### 3. FAISS 인덱스 불필요한 재빌드

**문제:** 매번 앱이 재시작될 때마다 전체 문서를 임베딩하여 FAISS를 새로 빌드하면, 문서 수가 많을수록 수 분 이상 소요되어 사용성이 크게 저하됩니다.

**해결:** Lakehouse의 MD5 해시 로그와 FAISS 저장 디렉토리를 비교하여 **해시가 동일하면 디스크에서 인덱스를 로드**하고, 변경이 감지된 경우에만 재빌드하도록 처리했습니다. Streamlit에서는 `@st.cache_resource`로 추가적으로 메모리 캐싱을 적용했습니다.

```python
if os.path.exists(faiss_dir) and current_meta == old_meta:
    return FAISS.load_local(faiss_dir, embeddings, ...)  # 캐시 로드
# 변경 있을 때만 재빌드
```

---

### 4. LLM 쿼리 분류 실패 시 시스템 중단

**문제:** 쿼리 유형 분류를 Gemini API 호출에 전적으로 의존할 경우, API 오류(Rate Limit, 네트워크 이슈 등)가 발생하면 모든 검색이 실패합니다.

**해결:** LLM 분류를 `try-except`로 감싸고, 실패 시 규칙 기반 fallback 로직이 즉시 동작하도록 이중 분기를 설계했습니다.

```python
try:
    result = llm.invoke(classify_prompt).content.strip().lower()
    if result in ('regex', 'word', 'sentence'):
        return result
except Exception:
    pass
# Fallback: 규칙 기반
if any(c in query for c in r'.^$*+?{}[]\\|()'):
    return 'regex'
return 'word' if len(query.split()) == 1 else 'sentence'
```

---

### 5. EasyOCR 지연 초기화 및 메모리 과부하

**문제:** EasyOCR 모델(`ko`, `en` 2개 언어)은 초기화 시 수백 MB의 메모리를 사용합니다. 이미지가 없는 데이터셋에서도 항상 로드하면 불필요한 리소스를 낭비합니다.

**해결:** 전역 변수와 지연 초기화(Lazy Initialization) 패턴을 사용하여 실제로 이미지 파일이 처음 처리될 때 한 번만 OCR 엔진을 초기화하도록 구현했습니다. Streamlit에서는 `@st.cache_resource`로 세션 내 재사용합니다.

```python
_ocr_reader = None
def get_ocr_reader():
    global _ocr_reader
    if _ocr_reader is None:
        _ocr_reader = easyocr.Reader(['ko', 'en'], gpu=False)
    return _ocr_reader
```

---

### 6. Streamlit 세션 내 대화 맥락 유지

**문제:** Streamlit은 사용자 인터랙션마다 전체 스크립트를 재실행하기 때문에, 일반 Python 변수로 대화 히스토리를 저장하면 매 응답 후 초기화됩니다.

**해결:** `st.session_state`에 `chat_history` 리스트를 저장하고, `reload_trigger` 카운터로 캐시 무효화를 제어합니다. 대화 기록은 슬라이딩 윈도우(최근 12 메시지)로 LLM에 전달하여 컨텍스트 길이를 제한합니다.

```python
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
window = st.session_state.chat_history[-12:]  # 슬라이딩 윈도우
```

---

### 7. BM25 + Semantic 결과 중복 병합

**문제:** BM25, Fuzzy, FAISS 3가지 검색이 같은 파일의 같은 청크를 반환할 경우, LLM 컨텍스트가 동일한 정보로 중복 채워져 다른 문서에 대한 정보가 밀려나는 문제가 발생합니다.

**해결:** 모든 검색 결과를 통합한 뒤, `(file_name, content[:100])` 튜플을 키로 하는 `seen` 집합을 통해 중복을 제거하여 다양한 출처의 문서가 LLM에 전달되도록 보장합니다.

```python
seen = set()
all_docs = []
for doc in extra_docs + semantic_docs:
    key = (doc.metadata.get('file_name', ''), doc.page_content[:100])
    if key not in seen:
        seen.add(key)
        all_docs.append(doc)
```

---
## ✅ 실행 예시 (Streamlit)

### ① ^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$ (정규식) 검색
<img width="696" height="734" alt="image" src="https://github.com/user-attachments/assets/a89888c5-4989-4b6f-9a2b-77ebe1be333a" />

### ② 딥오토 (단어) 검색
<img width="685" height="708" alt="image" src="https://github.com/user-attachments/assets/95415310-4b77-4a60-a1c2-1b24fbbd1d98" />


### ③ 재무제표 요약 (문장) 검색
<img width="703" height="707" alt="image" src="https://github.com/user-attachments/assets/5da4d585-d0e6-456a-81d5-19f749bc9a81" />


---

## 🪞 최종 회고 및 성찰

### 핵심 성과

**검색 전략의 적응적 라우팅이 핵심 가치를 만들었습니다.** 단일 검색 방식(벡터만, 키워드만)으로는 커버할 수 없는 다양한 쿼리 패턴을 AI가 자동으로 분류하고 라우팅하는 아이디어가 시스템의 실용성을 크게 높였습니다. 특히 정규식 검색(`\d{3}-\d{4}`처럼 전화번호 패턴 탐색)을 자연어 인터페이스 안에 자연스럽게 녹여낸 점이 독창적입니다.

**MD5 기반 증분 적재는 운영 효율을 극적으로 개선했습니다.** 수십 개의 파일 중 하나만 수정되었을 때 전체를 재처리하지 않는 설계 덕분에, 데이터 업데이트 비용이 O(n) → O(변경 파일 수)로 줄어들었습니다.

**이중 추출 전략(PyMuPDF + pdfplumber)이 데이터 품질을 높였습니다.** 표가 많은 재무 문서나 보고서에서 표 데이터까지 정확히 수집하여 LLM이 수치 기반 질문에도 정확히 답변할 수 있게 되었습니다.

### 한계점 및 개선 여지

**FAISS 인덱스의 런타임 저장 한계.** 현재 FAISS 인덱스를 Colab 로컬(`/content/`)에 저장하므로 런타임 재시작 시 재빌드가 필요합니다. Google Drive에 인덱스를 저장하거나, Pinecone/Weaviate 같은 영구 벡터 DB로 이전하면 이 문제를 해소할 수 있습니다.

**청크 분할 전략의 단순성.** `RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)`는 범용적이지만 문서 유형별로 최적화되어 있지 않습니다. PDF는 페이지 경계, DOCX는 단락, 표는 행 단위로 분할하는 의미론적 청킹(Semantic Chunking)을 적용하면 검색 정확도가 더 높아질 것입니다.

**LLM 비용 구조.** 쿼리 분류와 답변 생성에 각각 Gemini API를 호출하는 구조는 쿼리 빈도가 높아질수록 비용이 선형 증가합니다. 분류기는 경량 로컬 모델로 대체하거나, 동일 LLM 호출 내에서 분류와 답변을 한 번에 처리하는 방식으로 API 호출을 줄일 수 있습니다.

**멀티모달 한계.** 현재 이미지는 EasyOCR로 텍스트만 추출합니다. 차트, 다이어그램, 도표 이미지는 OCR로 의미 있는 텍스트를 추출하기 어려우므로, 향후 Gemini의 Vision 기능을 활용한 이미지 이해가 필요합니다.

---

## 🔮 향후 발전 계획

| 단계 | 계획 | 내용 |
|------|------|------|
| 단기 | 영구 벡터 스토리지 | FAISS 인덱스를 Google Drive에 저장하거나 Chroma DB로 교체 |
| 단기 | Re-Ranking 도입 | Cross-Encoder 기반 재순위화로 LLM 컨텍스트 관련성 향상 |
| 중기 | 멀티모달 확장 | Gemini Vision API로 차트·도표 이미지를 검색 가능하게 인덱싱 |
| 중기 | 스트리밍 응답 | Gemini Streaming + `st.write_stream`으로 실시간 답변 출력 |
| 장기 | 에이전트 분석 | LangChain Agent + Python REPL로 데이터 직접 연산·시각화 |
| 장기 | REST API 서버화 | FastAPI로 래핑하여 외부 서비스·챗봇 플랫폼과 연동 |

---

