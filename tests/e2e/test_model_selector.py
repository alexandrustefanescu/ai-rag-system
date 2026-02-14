"""E2E tests for the model selector dropdown."""

import pytest
from playwright.sync_api import Page, expect

pytestmark = pytest.mark.e2e


class TestModelSelector:
    def test_model_dropdown_visible(self, page: Page, base_url: str):
        page.goto(base_url)
        expect(page.locator("#model-select")).to_be_visible()

    def test_model_dropdown_populates(self, page: Page, base_url: str):
        page.goto(base_url)

        # Wait for status fetch to populate the dropdown
        page.wait_for_timeout(2000)

        select = page.locator("#model-select")
        options = select.locator("option")

        # Should have the 2 configured models
        expect(options).to_have_count(2)

    def test_model_options_contain_configured_models(self, page: Page, base_url: str):
        """
        Verifies the model selector contains exactly one option for each configured model.
        
        Checks that the #model-select element includes one option with value "gemma3:1b" and one option with value "llama3.2:1b".
        """
        page.goto(base_url)
        page.wait_for_timeout(2000)

        select = page.locator("#model-select")
        # Check that expected models are present
        for model in ["gemma3:1b", "llama3.2:1b"]:
            expect(select.locator(f"option[value='{model}']")).to_have_count(1)

    def test_default_model_selected(self, page: Page, base_url: str):
        """
        Check that the model selector defaults to "gemma3:1b" after the page loads.
        
        Asserts that the #model-select element's value is "gemma3:1b".
        """
        page.goto(base_url)
        page.wait_for_timeout(2000)

        expect(page.locator("#model-select")).to_have_value("gemma3:1b")

    def test_can_change_model(self, page: Page, base_url: str):
        """
        Changes the selected model in the model dropdown and verifies the selection updates.
        
        Selects the option with value "llama3.2:1b" in the `#model-select` element and asserts the element's value becomes "llama3.2:1b".
        """
        page.goto(base_url)
        page.wait_for_timeout(2000)

        page.select_option("#model-select", "llama3.2:1b")
        expect(page.locator("#model-select")).to_have_value("llama3.2:1b")