"""E2E tests for the chat interface."""

import pytest
from playwright.sync_api import Page, expect

pytestmark = pytest.mark.e2e


class TestPageLoad:
    def test_title(self, page: Page, base_url: str):
        """
        Verify the application page title is "RAG System".
        
        Parameters:
            base_url (str): URL of the application home page to navigate to before asserting the title.
        """
        page.goto(base_url)
        expect(page).to_have_title("RAG System")

    def test_header_visible(self, page: Page, base_url: str):
        """
        Verify the page header displays "RAG System".
        """
        page.goto(base_url)
        expect(page.locator("header h1")).to_have_text("RAG System")

    def test_empty_state_message(self, page: Page, base_url: str):
        page.goto(base_url)
        expect(page.locator("#empty-state")).to_contain_text(
            "Ask a question about your documents"
        )

    def test_input_and_send_button_visible(self, page: Page, base_url: str):
        """
        Verify the question input and send button are visible on the page.
        
        Navigates to the provided base URL and asserts that elements with selectors `#question` and `#send-btn` are visible.
        """
        page.goto(base_url)
        expect(page.locator("#question")).to_be_visible()
        expect(page.locator("#send-btn")).to_be_visible()

    def test_status_bar_shows_chunks(self, page: Page, base_url: str):
        """
        Verify the status bar displays the document chunk count.
        
        Navigates to the provided base URL and asserts that the element with id "doc-count" contains the text "chunks".
        """
        page.goto(base_url)
        expect(page.locator("#doc-count")).to_contain_text("chunks")


class TestChatInteraction:
    def test_send_question_shows_user_message(self, page: Page, base_url: str):
        """
        Verify that submitting a question causes the user's message to appear in the chat.
        
        Asserts that after sending "What is Python?" a `.message.user` element contains the same text.
        """
        page.goto(base_url)

        page.fill("#question", "What is Python?")
        page.click("#send-btn")

        user_msg = page.locator(".message.user")
        expect(user_msg).to_have_text("What is Python?")

    def test_send_question_shows_assistant_response(self, page: Page, base_url: str):
        """
        Verifies that submitting a question causes an assistant response to appear.
        
        Waits up to 10 seconds for an element with selector `.message.assistant` to become visible after sending the question.
        """
        page.goto(base_url)

        page.fill("#question", "What is Python?")
        page.click("#send-btn")

        assistant_msg = page.locator(".message.assistant")
        expect(assistant_msg).to_be_visible(timeout=10_000)

    def test_empty_state_removed_after_question(self, page: Page, base_url: str):
        """
        Verify that submitting a question removes the empty-state element from the page.
        
        Navigates to the provided base URL, sends a user question, and asserts that the element with id "empty-state" is no longer present.
        """
        page.goto(base_url)

        page.fill("#question", "Hello")
        page.click("#send-btn")

        expect(page.locator("#empty-state")).to_have_count(0)

    def test_enter_key_sends_question(self, page: Page, base_url: str):
        page.goto(base_url)

        page.fill("#question", "Test question")
        page.press("#question", "Enter")

        user_msg = page.locator(".message.user")
        expect(user_msg).to_have_text("Test question")

    def test_input_cleared_after_send(self, page: Page, base_url: str):
        """
        Verify that sending a question clears the input field.
        
        Navigates to the page, enters text into the `#question` input, clicks the send button, and asserts the input value is empty afterward.
        """
        page.goto(base_url)

        page.fill("#question", "Some question")
        page.click("#send-btn")

        expect(page.locator("#question")).to_have_value("")