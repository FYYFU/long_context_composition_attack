#!/usr/bin/env bash
RELEASE_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RELEASE_ROOT="$(cd "${RELEASE_SCRIPT_DIR}/../../.." && pwd)"
cd "${RELEASE_SCRIPT_DIR}"
mkdir -p logs
# Download multilingual corpora.
# Include language filtering to keep text mostly in the target language.

cd "${RELEASE_SCRIPT_DIR}"

echo "========================================"
echo "下载多语言 Wikipedia 语料库"
echo "========================================"
echo "语言: 中文(zh), 俄语(ru), 越南语(vi), 孟加拉语(bn), 斯瓦希里语(sw)"
echo "启用语言过滤: 最小比例 50%"
echo "========================================"

# Download each language separately for easier control.
for lang in ar pt; do
    echo ""
    echo "========================================"
    echo "下载 ${lang} 语料库..."
    echo "========================================"
    
    python download_multilingual_corpus.py download \
        --language "${lang}" \
        --output_dir "./noise_datasets" \
        --max_samples 10000 \
        --filter_language True \
        --min_language_ratio 0.9
done

echo ""
echo "完成!"
echo "语料库保存在 ./noise_datasets/"
echo "  - wikipedia_zh/ (中文)"
echo "  - wikipedia_ru/ (俄语)"
echo "  - wikipedia_vi/ (越南语)"
echo "  - wikipedia_bn/ (孟加拉语)"
echo "  - wikipedia_sw/ (斯瓦希里语)"
echo "  - wikipedia_ar/ (阿拉伯语)"
echo "  - wikipedia_pt/ (葡萄牙语)"
