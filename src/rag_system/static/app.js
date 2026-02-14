const chat = document.getElementById("chat");
const input = document.getElementById("question");
const sendBtn = document.getElementById("send-btn");
const uploadBtn = document.getElementById("upload-btn");
const modelSelect = document.getElementById("model-select");
const docCount = document.getElementById("doc-count");
const docList = document.getElementById("doc-list");
const docPanel = document.getElementById("doc-panel");
let emptyState = document.getElementById("empty-state");
let lastQuestion = "";

/**
 * Scrolls the chat container element to its bottom.
 */
function scrollToBottom() {
    chat.scrollTop = chat.scrollHeight;
}

/**
 * Remove the empty-state element from the DOM and clear its reference.
 *
 * Clears the global `emptyState` variable after removing the element if it exists.
 */
function clearEmptyState() {
    if (emptyState) {
        emptyState.remove();
        emptyState = null;
    }
}

/**
 * Enable or disable user input controls to prevent interaction during asynchronous operations.
 *
 * @param {boolean} locked - If `true`, disable the send button and model selector; if `false`, enable them.
 */
function setInputLock(locked) {
    sendBtn.disabled = locked;
    modelSelect.disabled = locked;
}

/**
 * Convert a string so it can be safely inserted into HTML by escaping special characters.
 * @param {string} str - The input string to escape.
 * @returns {string} The input with special characters replaced by HTML entities suitable for assigning to element.innerHTML.
 */
function escapeHtml(str) {
    const div = document.createElement("div");
    div.textContent = str;
    return div.innerHTML;
}

/**
 * Append a chat message to the chat container and scroll to the bottom.
 *
 * If `role` is "assistant", the message is rendered with an answer section and
 * an optional expandable list of sources; each source includes its name,
 * relevance score, and a truncated excerpt.
 *
 * @param {string} role - Message role; "assistant" renders rich HTML with sources, other values render plain text.
 * @param {string} content - Message text content.
 * @param {Array<{source: string, relevance: number, text: string}>} [sources] - Optional array of source objects used when `role` is "assistant".
 */
function addMessage(role, content, sources) {
    clearEmptyState();

    const div = document.createElement("div");
    div.className = "message " + role;

    if (role === "assistant") {
        let html = '<div class="answer">' + escapeHtml(content) + "</div>";
        if (sources && sources.length > 0) {
            html += '<details class="sources"><summary>Sources (' + sources.length + ")</summary>";
            sources.forEach(function(s) {
                html += '<div class="source-item">';
                html += '<div class="meta">' + escapeHtml(s.source) + " &middot; relevance: " + s.relevance.toFixed(2) + "</div>";
                html += escapeHtml(s.text.substring(0, 200));
                if (s.text.length > 200) html += "...";
                html += "</div>";
            });
            html += "</details>";
        }
        div.innerHTML = html;
    } else {
        div.textContent = content;
    }

    chat.appendChild(div);
    scrollToBottom();
}

/**
 * Displays an assistant-styled error message in the chat with a Retry button.
 *
 * The message is appended to the chat UI and a Retry button is included that,
 * when clicked, re-invokes askQuestion using the last submitted question.
 *
 * @param {string} content - The error text to display in the message.
 */
function addErrorMessage(content) {
    clearEmptyState();

    const div = document.createElement("div");
    div.className = "message assistant error";
    div.id = "error-msg";

    const answer = document.createElement("div");
    answer.className = "answer";
    answer.textContent = content;
    div.appendChild(answer);

    const btn = document.createElement("button");
    btn.className = "retry-btn";
    btn.textContent = "Retry";
    btn.addEventListener("click", function() { askQuestion(lastQuestion); });
    div.appendChild(btn);

    chat.appendChild(div);
    scrollToBottom();
}

/**
 * Display a transient "thinking" indicator in the chat UI.
 *
 * Clears any empty-state element, appends a thinking indicator element to the chat container,
 * and scrolls the chat to the bottom so the indicator is visible.
 */
function showThinking() {
    clearEmptyState();
    const div = document.createElement("div");
    div.className = "thinking";
    div.id = "thinking";
    div.innerHTML = '<span class="spinner"></span> Thinking...';
    chat.appendChild(div);
    scrollToBottom();
}

/**
 * Remove the thinking indicator element from the chat UI.
 *
 * If the element with id "thinking" is not present, this function does nothing.
 */
function hideThinking() {
    const el = document.getElementById("thinking");
    if (el) el.remove();
}

/**
 * Send a question to the backend and render the assistant's response or an error in the chat UI.
 *
 * If `retryText` is provided the function will use it as the question, remove any existing error message,
 * and avoid adding a duplicate user message; otherwise it reads and clears the input field and adds the user message.
 * The UI is locked while the request is in flight and a thinking indicator is shown.
 * @param {string} [retryText] - Optional question text to retry instead of using the current input value.
 */
async function askQuestion(retryText) {
    const q = retryText || input.value.trim();
    if (!q) return;

    if (!retryText) input.value = "";
    lastQuestion = q;

    if (retryText) {
        const errMsg = document.getElementById("error-msg");
        if (errMsg) errMsg.remove();
    }

    setInputLock(true);
    if (!retryText) addMessage("user", q);
    showThinking();

    try {
        const res = await fetch("/api/v1/ask", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ question: q, model: modelSelect.value }),
        });
        if (!res.ok) throw new Error(res.statusText);
        const data = await res.json();
        addMessage("assistant", data.answer, data.sources);
    } catch (err) {
        addErrorMessage("Error: Could not reach the server.");
    } finally {
        hideThinking();
        setInputLock(false);
        input.focus();
    }
}

/**
 * Uploads selected files to the server and updates the UI with the result.
 *
 * Sends the files chosen in the file input to the /api/v1/upload endpoint, updates the upload button text and disabled state while uploading, and refreshes server status on success. On failure the button shows an error state. The file input is cleared and the upload button is reset after three seconds.
 */
async function uploadFiles() {
    const fileInput = document.getElementById("file-input");
    const files = fileInput.files;
    if (!files.length) return;

    uploadBtn.disabled = true;
    uploadBtn.textContent = "Uploading...";

    const formData = new FormData();
    for (let i = 0; i < files.length; i++) {
        formData.append("files", files[i]);
    }

    try {
        const res = await fetch("/api/v1/upload", {
            method: "POST",
            body: formData,
        });
        if (!res.ok) throw new Error(res.statusText);
        const data = await res.json();
        uploadBtn.textContent = data.status === "ok"
            ? data.files_saved + " file(s), " + data.chunks + " chunks!"
            : "No valid files (.txt, .md, .pdf)";
        fetchStatus();
    } catch (err) {
        uploadBtn.textContent = "Upload failed";
    }

    fileInput.value = "";
    setTimeout(function() {
        uploadBtn.textContent = "Upload Documents";
        uploadBtn.disabled = false;
    }, 3000);
}

/**
 * Refreshes application status from the server and updates the UI.
 *
 * If the model selector is still showing the loading placeholder, replaces its options
 * with the server-provided available models and selects the active model. Updates the
 * document count element with the reported number of chunks. Network or parsing errors
 * are ignored.
 */
async function fetchStatus() {
    try {
        const res = await fetch("/api/v1/status");
        if (!res.ok) return;
        const data = await res.json();

        if (data.available_models && modelSelect.options.length <= 1 && modelSelect.options[0].text === "Loading...") {
            modelSelect.innerHTML = "";
            data.available_models.forEach(function(m) {
                const opt = document.createElement("option");
                opt.value = m;
                opt.textContent = m;
                if (m === data.model) opt.selected = true;
                modelSelect.appendChild(opt);
            });
        }

        docCount.textContent = data.documents + " chunks";
    } catch (err) {
        // ignore — status will refresh on next action
    }
}

/**
 * Toggle the document panel's open state.
 *
 * If the panel is opened, fetches and renders the uploaded documents for display.
 */
function toggleDocPanel() {
    docPanel.classList.toggle("open");
    if (docPanel.classList.contains("open")) loadDocuments();
}

/**
 * Load uploaded documents from the server and render them into the document list panel.
 *
 * Fetches document metadata from /api/v1/documents, shows a loading placeholder while fetching,
 * then renders each document as an item with filename, size/chunk information, and a "Delete"
 * button that calls deleteDocument(filename). If no documents exist, shows an empty message;
 * on fetch failure, shows an error message.
 */
async function loadDocuments() {
    docList.innerHTML = '<div class="panel-empty">Loading...</div>';

    try {
        const res = await fetch("/api/v1/documents");
        if (!res.ok) throw new Error(res.statusText);
        const data = await res.json();

        if (data.files.length === 0) {
            docList.innerHTML = '<div class="panel-empty">No documents uploaded yet.</div>';
            return;
        }

        docList.innerHTML = "";
        data.files.forEach(function(f) {
            const item = document.createElement("div");
            item.className = "doc-item";

            const info = document.createElement("div");
            info.className = "doc-info";

            const name = document.createElement("div");
            name.className = "doc-name";
            name.textContent = f.filename;

            const size = document.createElement("div");
            size.className = "doc-size";
            size.textContent = f.size_kb > 0
                ? f.size_kb + " KB \u00b7 " + f.chunk_count + " chunks"
                : f.chunk_count + " chunks (file removed)";

            info.appendChild(name);
            info.appendChild(size);

            const btn = document.createElement("button");
            btn.className = "btn danger";
            btn.textContent = "Delete";
            btn.addEventListener("click", function() { deleteDocument(f.filename); });

            item.appendChild(info);
            item.appendChild(btn);
            docList.appendChild(item);
        });
    } catch (err) {
        docList.innerHTML = '<div class="panel-empty">Failed to load documents.</div>';
    }
}

/**
 * Prompt to delete a document, remove it from the server, and refresh UI state.
 *
 * Prompts the user to confirm deletion of the given file; if confirmed, sends a DELETE request for that filename,
 * reloads the document list and status on success, and shows an alert on failure.
 * @param {string} filename - The document filename to delete.
 */
async function deleteDocument(filename) {
    if (!confirm("Delete " + filename + "? The vector store will be re-indexed.")) return;

    try {
        const res = await fetch("/api/v1/documents/" + encodeURIComponent(filename), {
            method: "DELETE",
        });
        if (!res.ok) throw new Error(res.statusText);
        const data = await res.json();
        if (data.status === "ok") {
            loadDocuments();
            fetchStatus();
        }
    } catch (err) {
        alert("Delete failed.");
    }
}

fetchStatus();