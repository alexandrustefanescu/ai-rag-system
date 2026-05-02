"""Integration tests — end-to-end ingestion + retrieval pipeline."""

import pytest

from rag_system import vector_store as vs
from rag_system.config import VectorStoreConfig
from rag_system.document_loader import load_documents
from rag_system.models import Chunk
from rag_system.text_chunker import chunk_documents


@pytest.fixture
def tmp_vs_config(tmp_path):
    """Return a VectorStoreConfig using a temporary directory."""
    return VectorStoreConfig(db_path=str(tmp_path / "chroma_db"))


class TestIngestionPipeline:
    """Test the full ingest path: load -> chunk -> store."""

    def test_txt_ingestion_and_retrieval(self, tmp_vs_config, tmp_path):
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()
        (docs_dir / "python.txt").write_text(
            "Python is a high-level programming language. "
            "It supports object-oriented and functional programming.",
            encoding="utf-8",
        )
        (docs_dir / "java.txt").write_text(
            "Java is a compiled language that runs on the JVM. "
            "It is statically typed and widely used in enterprises.",
            encoding="utf-8",
        )

        documents = load_documents(str(docs_dir))
        assert len(documents) == 2

        chunks = chunk_documents(documents)
        assert len(chunks) == 2

        client = vs.get_client(tmp_vs_config)
        collection = vs.reset_collection(client, tmp_vs_config)
        vs.add_chunks(collection, chunks)
        assert collection.count() == 2

        results = vs.query(collection, "programming language", n_results=2)
        docs = results["documents"][0]
        assert len(docs) == 2

    def test_markdown_ingestion(self, tmp_vs_config, tmp_path):
        docs_dir = tmp_path / "mddocs"
        docs_dir.mkdir()
        (docs_dir / "guide.md").write_text(
            "# Machine Learning Guide\n\n"
            "Machine learning is a subset of artificial intelligence.\n\n"
            "## Supervised Learning\n\n"
            "Uses labeled data to train models.",
            encoding="utf-8",
        )

        documents = load_documents(str(docs_dir))
        assert len(documents) == 1
        assert "Machine" in documents[0].content

        chunks = chunk_documents(documents)
        client = vs.get_client(tmp_vs_config)
        collection = vs.reset_collection(client, tmp_vs_config)
        vs.add_chunks(collection, chunks)

        results = vs.query(collection, "artificial intelligence", n_results=1)
        assert len(results["documents"][0]) == 1

    def test_mixed_file_types(self, tmp_vs_config, tmp_path):
        docs_dir = tmp_path / "mixed"
        docs_dir.mkdir()
        (docs_dir / "notes.txt").write_text(
            "Plain text notes about databases.", encoding="utf-8"
        )
        (docs_dir / "readme.md").write_text(
            "# README\n\nProject documentation.", encoding="utf-8"
        )
        (docs_dir / "ignore.csv").write_text("a,b,c\n1,2,3", encoding="utf-8")

        documents = load_documents(str(docs_dir))
        assert len(documents) == 2  # csv should be skipped

        chunks = chunk_documents(documents)
        client = vs.get_client(tmp_vs_config)
        collection = vs.reset_collection(client, tmp_vs_config)
        vs.add_chunks(collection, chunks)
        assert collection.count() == 2

    def test_reset_clears_previous_data(self, tmp_vs_config):
        client = vs.get_client(tmp_vs_config)

        coll = vs.reset_collection(client, tmp_vs_config)
        vs.add_chunks(coll, [Chunk(text="old data", metadata={"source": "old.txt"})])
        assert coll.count() == 1

        coll = vs.reset_collection(client, tmp_vs_config)
        assert coll.count() == 0

        vs.add_chunks(coll, [Chunk(text="new data", metadata={"source": "new.txt"})])
        assert coll.count() == 1

        results = vs.query(coll, "data", n_results=5)
        sources = [m["source"] for m in results["metadatas"][0]]
        assert "new.txt" in sources
        assert "old.txt" not in sources

    def test_semantic_similarity_ranking(self, tmp_vs_config):
        client = vs.get_client(tmp_vs_config)
        collection = vs.reset_collection(client, tmp_vs_config)

        chunks = [
            Chunk(
                text="Cats are fluffy domesticated animals that purr.",
                metadata={"source": "doc0.txt"},
            ),
            Chunk(
                text="The stock market experienced a downturn in Q3.",
                metadata={"source": "doc1.txt"},
            ),
            Chunk(
                text="Dogs are loyal pets that love to play fetch.",
                metadata={"source": "doc2.txt"},
            ),
        ]
        vs.add_chunks(collection, chunks)

        results = vs.query(collection, "pets and animals", n_results=3)
        docs = results["documents"][0]

        # Animal-related docs should rank higher (lower distance)
        assert "Cats" in docs[0] or "Dogs" in docs[0]
        # Stock market doc should be last
        assert "stock" in docs[-1].lower()
