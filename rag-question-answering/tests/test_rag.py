from src.rag import LocalRAG


def test_split_text_preserves_content():
    text = 'This is a short sentence. ' * 100
    chunks = LocalRAG.split_text(text, chunk_size=120, overlap=20)
    assert len(chunks) > 1
    assert all(chunks)
    assert chunks[0][:10] == 'This is a '


def test_split_text_rejects_bad_overlap():
    try:
        LocalRAG.split_text('hello world', chunk_size=10, overlap=10)
    except ValueError:
        return
    raise AssertionError('Expected ValueError for overlap >= chunk size')


def test_empty_text_returns_no_chunks():
    assert LocalRAG.split_text('   ') == []
