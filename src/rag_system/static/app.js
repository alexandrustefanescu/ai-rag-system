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

function scrollToBottom() {
    chat.scrollTop = chat.scrollHeight;
}

function clearEmptyState() {
    if (emptyState) {
        emptyState.remove();
        emptyState = null;
    }
}

function setInputLock(locked) {
    sendBtn.disabled = locked;
    modelSelect.disabled = locked;
}

function escapeHtml(str) {
    const div = document.createElement("div");
    div.textContent = str;
    return div.innerHTML;
}

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

function showThinking() {
    clearEmptyState();
    const div = document.createElement("div");
    div.className = "thinking";
    div.id = "thinking";
    div.innerHTML = '<span class="spinner"></span> Thinking...';
    chat.appendChild(div);
    scrollToBottom();
}

function hideThinking() {
    const el = document.getElementById("thinking");
    if (el) el.remove();
}

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

function toggleDocPanel() {
    docPanel.classList.toggle("open");
    if (docPanel.classList.contains("open")) loadDocuments();
}

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
