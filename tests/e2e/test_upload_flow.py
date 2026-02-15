"""E2E tests for file upload functionality."""

import re
from pathlib import Path

import pytest
from playwright.sync_api import Page, expect

pytestmark = pytest.mark.e2e


class TestUpload:
    def test_upload_button_visible(self, page: Page, base_url: str):
        page.goto(base_url)
        expect(page.locator("#upload-btn")).to_be_visible()

    def test_upload_txt_file(
        self,
        page: Page,
        base_url: str,
        tmp_path: Path,
    ):
        page.goto(base_url)

        # Create a test file
        test_file = tmp_path / "test_doc.txt"
        test_file.write_text(
            "Python is a high-level programming language "
            "used for web development and data science."
        )

        # Upload via the hidden file input
        page.locator("#file-input").set_input_files(str(test_file))

        # Upload button should show feedback then reset
        btn = page.locator("#upload-btn")
        expect(btn).to_contain_text("chunks!", timeout=10_000)
        expect(btn).to_have_text(
            "Upload Documents",
            timeout=10_000,
        )

    def test_chunk_count_updates_after_upload(
        self,
        page: Page,
        base_url: str,
        tmp_path: Path,
    ):
        page.goto(base_url)

        test_file = tmp_path / "sample.txt"
        test_file.write_text("Some content for testing chunk count updates.")

        page.locator("#file-input").set_input_files(str(test_file))

        # Wait for upload to complete and status to refresh
        page.wait_for_timeout(2000)

        doc_count = page.locator("#doc-count")
        expect(doc_count).to_contain_text("chunks")

    def test_uploaded_file_appears_in_manage_panel(
        self,
        page: Page,
        base_url: str,
        tmp_path: Path,
    ):
        page.goto(base_url)

        test_file = tmp_path / "visible_doc.txt"
        test_file.write_text("Content that should appear in the manage panel.")

        page.locator("#file-input").set_input_files(str(test_file))
        page.wait_for_timeout(2000)

        # Open manage panel
        page.get_by_text("Manage").click()
        expect(page.locator("#doc-panel")).to_have_class(
            re.compile("open"),
        )

        # The file should be listed
        expect(
            page.locator(".doc-name").filter(has_text="visible_doc.txt")
        ).to_be_visible()
