"""E2E tests for the document management panel."""

import re
from pathlib import Path

import pytest
from playwright.sync_api import Page, expect

pytestmark = pytest.mark.e2e


class TestDocumentPanel:
    def test_manage_button_opens_panel(self, page: Page, base_url: str):
        """
        Verify that clicking the "Manage" control opens the document panel.
        
        Asserts that the element with id "doc-panel" has a class matching "open".
        """
        page.goto(base_url)
        page.get_by_text("Manage").click()
        expect(page.locator("#doc-panel")).to_have_class(re.compile("open"))

    def test_close_button_closes_panel(self, page: Page, base_url: str):
        page.goto(base_url)
        page.get_by_text("Manage").click()
        expect(page.locator("#doc-panel")).to_have_class(re.compile("open"))

        page.locator("#doc-panel").get_by_text("Close").click()
        expect(page.locator("#doc-panel")).not_to_have_class(re.compile("open"))

    def test_empty_panel_message(self, page: Page, base_url: str):
        """
        Check that the document management panel displays an empty-state message when no documents are present.
        
        Opens the panel and asserts an element with class "panel-empty" is visible within 5 seconds.
        """
        page.goto(base_url)
        page.get_by_text("Manage").click()

        expect(page.locator(".panel-empty")).to_be_visible(timeout=5_000)

    def test_panel_lists_uploaded_documents(
        self,
        page: Page,
        base_url: str,
        tmp_path: Path,
    ):
        """
        Verify that a file uploaded via the file input appears in the document management panel with the expected filename and a size indicator containing "chunks".
        
        This test uploads a temporary file, opens the panel, and asserts the uploaded document is visible, its name contains "panel_test.txt", and its size text contains "chunks".
        """
        page.goto(base_url)

        # Upload a file first
        test_file = tmp_path / "panel_test.txt"
        test_file.write_text("Document for panel listing test.")
        page.locator("#file-input").set_input_files(str(test_file))
        page.wait_for_timeout(2000)

        # Open panel and verify
        page.get_by_text("Manage").click()

        doc_item = page.locator(".doc-item")
        expect(doc_item).to_be_visible(timeout=5_000)
        expect(page.locator(".doc-name")).to_contain_text("panel_test.txt")
        expect(page.locator(".doc-size")).to_contain_text("chunks")

    def test_delete_document(self, page: Page, base_url: str, tmp_path: Path):
        """
        End-to-end test that uploads a document, deletes it from the document panel, and verifies its removal.
        
        Creates a temporary file named `to_delete.txt`, uploads it via the page file input, opens the document management panel, accepts the confirmation dialog, clicks the delete control for that specific document, and asserts the document is no longer listed.
        """
        page.goto(base_url)

        # Upload a file
        test_file = tmp_path / "to_delete.txt"
        test_file.write_text("This file will be deleted.")
        page.locator("#file-input").set_input_files(str(test_file))
        page.wait_for_timeout(2000)

        # Open panel
        page.get_by_text("Manage").click()
        delete_item = page.locator(".doc-name").filter(has_text="to_delete.txt")
        expect(delete_item).to_be_visible(timeout=5_000)

        # Accept the confirmation dialog
        page.on("dialog", lambda dialog: dialog.accept())

        # Click the delete button for this specific document
        delete_item.locator("..").locator("..").locator(".btn.danger").click()
        page.wait_for_timeout(2000)

        # File should be removed
        expect(
            page.locator(".doc-name").filter(has_text="to_delete.txt")
        ).to_have_count(0)