/* ── DOM refs ────────────────────────────────────────────────── */

const chat = document.getElementById("chat");
const input = document.getElementById("question");
const sendBtn = document.getElementById("send-btn");
const uploadBtn = document.getElementById("upload-btn");
const modelSelect = document.getElementById("model-select");
const docCount = document.getElementById("doc-count");
const docList = document.getElementById("doc-list");
const docPanel = document.getElementById("doc-panel");
const modelPanel = document.getElementById("model-panel");
const modelList = document.getElementById("model-list");
const sidebar = document.getElementById("sidebar");
const sidebarList = document.getElementById("sidebar-list");

let emptyState = document.getElementById("empty-state");
let lastQuestion = "";

/* ── Multi-conversation storage ─────────────────────────────── */

var CONVOS_KEY = "rag_conversations";
var ACTIVE_KEY = "rag_active_chat";

function loadConversations() {
    try {
        return JSON.parse(
            localStorage.getItem(CONVOS_KEY)
        ) || {};
    } catch (e) {
        return {};
    }
}

function saveConversations(convos) {
    try {
        localStorage.setItem(
            CONVOS_KEY, JSON.stringify(convos)
        );
    } catch (e) {
        // storage full — silently ignore
    }
}

function getActiveId() {
    return localStorage.getItem(ACTIVE_KEY) || null;
}

function setActiveId(id) {
    if (id) {
        localStorage.setItem(ACTIVE_KEY, id);
    } else {
        localStorage.removeItem(ACTIVE_KEY);
    }
}

function generateId() {
    return Date.now().toString(36)
        + Math.random().toString(36).slice(2, 7);
}

function createConversation() {
    var convos = loadConversations();
    var id = generateId();
    convos[id] = {
        id: id,
        title: "New chat",
        messages: [],
        createdAt: Date.now()
    };
    saveConversations(convos);
    setActiveId(id);
    return id;
}

function deleteConversation(id) {
    var convos = loadConversations();
    delete convos[id];
    saveConversations(convos);

    if (getActiveId() === id) {
        // Switch to most recent remaining, or null.
        var remaining = Object.values(convos);
        remaining.sort(function(a, b) {
            return b.createdAt - a.createdAt;
        });
        if (remaining.length > 0) {
            switchToChat(remaining[0].id);
        } else {
            setActiveId(null);
            clearChatDom();
        }
    }
    renderSidebar();
}

function appendMessage(id, role, content, sources) {
    var convos = loadConversations();
    var convo = convos[id];
    if (!convo) return;

    convo.messages.push({
        role: role,
        content: content,
        sources: sources || null
    });

    // Auto-title from first user message.
    if (role === "user" && convo.title === "New chat") {
        convo.title = content.length > 40
            ? content.substring(0, 40) + "..."
            : content;
    }

    saveConversations(convos);
    renderSidebar();
}

function getMessages(id) {
    var convos = loadConversations();
    var convo = convos[id];
    return convo ? convo.messages : [];
}

/* ── Sidebar rendering ──────────────────────────────────────── */

function renderSidebar() {
    var convos = loadConversations();
    var list = Object.values(convos);
    list.sort(function(a, b) {
        return b.createdAt - a.createdAt;
    });

    var activeId = getActiveId();
    sidebarList.innerHTML = "";

    if (list.length === 0) {
        sidebarList.innerHTML =
            '<div class="sidebar-empty">'
            + "No conversations yet.<br>"
            + 'Click "+ New" to start one.'
            + "</div>";
        return;
    }

    list.forEach(function(c) {
        var item = document.createElement("div");
        item.className = "sidebar-item"
            + (c.id === activeId ? " active" : "");
        item.addEventListener("click", function(e) {
            if (e.target.closest(
                ".sidebar-item-delete"
            )) return;
            switchToChat(c.id);
        });

        var title = document.createElement("span");
        title.className = "sidebar-item-title";
        title.textContent = c.title;

        var del = document.createElement("button");
        del.className = "sidebar-item-delete";
        del.textContent = "\u2715";
        del.title = "Delete conversation";
        del.addEventListener("click", function(e) {
            e.stopPropagation();
            if (!confirm("Delete this conversation?")) {
                return;
            }
            deleteConversation(c.id);
        });

        item.appendChild(title);
        item.appendChild(del);
        sidebarList.appendChild(item);
    });
}

/* ── Chat switching ─────────────────────────────────────────── */

function clearChatDom() {
    chat.querySelectorAll(
        ".message, .thinking"
    ).forEach(function(el) { el.remove(); });

    // Remove old empty state if present.
    if (emptyState) {
        emptyState.remove();
        emptyState = null;
    }

    // Recreate empty state.
    var es = document.createElement("div");
    es.className = "empty-state";
    es.id = "empty-state";
    es.innerHTML =
        "Ask a question about your documents.<br>"
        + "Upload .txt, .md, or .pdf files "
        + "to get started.";
    chat.appendChild(es);
    emptyState = es;
}

function switchToChat(id) {
    setActiveId(id);
    clearChatDom();

    var msgs = getMessages(id);
    msgs.forEach(function(msg) {
        addMessage(
            msg.role, msg.content, msg.sources, true
        );
    });

    renderSidebar();
    input.focus();
}

function newChat() {
    var id = createConversation();
    clearChatDom();
    renderSidebar();
    input.focus();
    return id;
}

function toggleSidebar() {
    sidebar.classList.toggle("open");
}

/* ── Migrate old single-chat history ────────────────────────── */

function migrateOldHistory() {
    var old = null;
    try {
        old = JSON.parse(
            localStorage.getItem("rag_chat_history")
        );
    } catch (e) {
        return;
    }
    if (!old || !old.length) return;

    var id = generateId();
    var title = "New chat";
    for (var i = 0; i < old.length; i++) {
        if (old[i].role === "user") {
            var t = old[i].content;
            title = t.length > 40
                ? t.substring(0, 40) + "..." : t;
            break;
        }
    }

    var convos = loadConversations();
    convos[id] = {
        id: id,
        title: title,
        messages: old,
        createdAt: Date.now()
    };
    saveConversations(convos);
    setActiveId(id);
    localStorage.removeItem("rag_chat_history");
}

/* ── Textarea auto-resize & Enter handling ──────────────────── */

input.addEventListener("keydown", function(e) {
    if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        askQuestion();
    }
});

input.addEventListener("input", function() {
    this.style.height = "auto";
    this.style.height =
        Math.min(this.scrollHeight, 160) + "px";
});

/* ── Markdown rendering ─────────────────────────────────────── */

if (typeof marked !== "undefined") {
    marked.setOptions({ breaks: true, gfm: true });
}

function addCopyButtons(container) {
    container.querySelectorAll("pre").forEach(
        function(pre) {
            var wrapper = document.createElement("div");
            wrapper.className = "code-block";
            pre.parentNode.insertBefore(wrapper, pre);
            wrapper.appendChild(pre);

            var btn = document.createElement("button");
            btn.className = "copy-btn";
            btn.textContent = "Copy";
            btn.addEventListener("click", function() {
                var code = pre.querySelector("code");
                var text = (code || pre).textContent;
                navigator.clipboard.writeText(text)
                    .then(function() {
                        btn.textContent = "Copied!";
                        setTimeout(function() {
                            btn.textContent = "Copy";
                        }, 2000);
                    });
            });
            wrapper.appendChild(btn);
        }
    );
}

function renderMarkdown(text) {
    if (typeof marked !== "undefined") {
        return marked.parse(text);
    }
    return escapeHtml(text);
}

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
    var div = document.createElement("div");
    div.textContent = str;
    return div.innerHTML;
}

function addMessage(role, content, sources, skipSave) {
    clearEmptyState();

    var div = document.createElement("div");
    div.className = "message " + role;

    if (role === "assistant") {
        var html = '<div class="answer">'
            + renderMarkdown(content) + "</div>";
        if (sources && sources.length > 0) {
            html += '<details class="sources">'
                + "<summary>Sources ("
                + sources.length + ")</summary>";
            sources.forEach(function(s) {
                html += '<div class="source-item">';
                html += '<div class="meta">'
                    + escapeHtml(s.source)
                    + " &middot; relevance: "
                    + s.relevance.toFixed(2)
                    + "</div>";
                html += escapeHtml(
                    s.text.substring(0, 200)
                );
                if (s.text.length > 200) html += "...";
                html += "</div>";
            });
            html += "</details>";
        }
        div.innerHTML = html;
        addCopyButtons(div);
    } else {
        div.textContent = content;
    }

    chat.appendChild(div);
    scrollToBottom();

    if (!skipSave) {
        var activeId = getActiveId();
        if (!activeId) {
            activeId = newChat();
        }
        appendMessage(activeId, role, content, sources);
    }
}

function addErrorMessage(content) {
    clearEmptyState();

    var div = document.createElement("div");
    div.className = "message assistant error";
    div.id = "error-msg";

    var answer = document.createElement("div");
    answer.className = "answer";
    answer.textContent = content;
    div.appendChild(answer);

    var btn = document.createElement("button");
    btn.className = "retry-btn";
    btn.textContent = "Retry";
    btn.addEventListener("click", function() {
        askQuestion(lastQuestion);
    });
    div.appendChild(btn);

    chat.appendChild(div);
    scrollToBottom();
}

function showThinking() {
    clearEmptyState();
    var div = document.createElement("div");
    div.className = "thinking";
    div.id = "thinking";
    div.innerHTML =
        '<span class="spinner"></span> Thinking...';
    chat.appendChild(div);
    scrollToBottom();
}

function hideThinking() {
    var el = document.getElementById("thinking");
    if (el) el.remove();
}

async function askQuestion(retryText) {
    var q = retryText || input.value.trim();
    if (!q) return;

    if (!retryText) {
        input.value = "";
        input.style.height = "auto";
    }
    lastQuestion = q;

    if (retryText) {
        var errMsg = document.getElementById("error-msg");
        if (errMsg) errMsg.remove();
    }

    // Ensure we have an active conversation.
    if (!getActiveId()) {
        newChat();
    }

    setInputLock(true);
    if (!retryText) addMessage("user", q);
    showThinking();

    try {
        var res = await fetch("/api/v1/ask", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                question: q,
                model: modelSelect.value
            }),
        });
        if (!res.ok) {
            var data = await res.json().catch(
                function() { return {}; }
            );
            throw new Error(
                data.detail || res.statusText
            );
        }
        var data = await res.json();
        addMessage(
            "assistant", data.answer, data.sources
        );
    } catch (err) {
        addErrorMessage("Error: " + err.message);
    } finally {
        hideThinking();
        setInputLock(false);
        input.focus();
    }
}

async function uploadFiles() {
    var fileInput = document.getElementById("file-input");
    var files = fileInput.files;
    if (!files.length) return;

    // Ensure the doc panel is open so the user sees
    // the result.
    if (!docPanel.classList.contains("open")) {
        toggleDocPanel();
    }

    uploadBtn.disabled = true;
    uploadBtn.textContent = "Uploading...";

    var formData = new FormData();
    for (var i = 0; i < files.length; i++) {
        formData.append("files", files[i]);
    }

    try {
        var res = await fetch("/api/v1/upload", {
            method: "POST",
            body: formData,
        });
        if (!res.ok) throw new Error(res.statusText);
        var data = await res.json();
        uploadBtn.textContent =
            data.status === "ok"
                ? data.files_saved + " added!"
                : "No valid files";
        fetchStatus();
        loadDocuments();
    } catch (err) {
        uploadBtn.textContent = "Failed";
    }

    fileInput.value = "";
    setTimeout(function() {
        uploadBtn.textContent = "+ Add";
        uploadBtn.disabled = false;
    }, 3000);
}

async function fetchStatus() {
    try {
        var res = await fetch("/api/v1/status");
        if (!res.ok) return;
        var data = await res.json();

        var downloaded = data.downloaded_models || [];
        modelSelect.innerHTML = "";

        if (downloaded.length === 0) {
            var opt = document.createElement("option");
            opt.value = "";
            opt.textContent = "No models downloaded";
            opt.disabled = true;
            opt.selected = true;
            modelSelect.appendChild(opt);
        } else {
            downloaded.forEach(function(m) {
                var opt =
                    document.createElement("option");
                opt.value = m;
                opt.textContent = m;
                if (m === data.model) {
                    opt.selected = true;
                }
                modelSelect.appendChild(opt);
            });
        }

        docCount.textContent =
            data.documents + " chunks";
    } catch (err) {
        // status will refresh on next action
    }
}

function toggleDocPanel() {
    modelPanel.classList.remove("open");
    docPanel.classList.toggle("open");
    if (docPanel.classList.contains("open")) {
        loadDocuments();
    }
}

async function loadDocuments() {
    docList.innerHTML =
        '<div class="panel-empty">Loading...</div>';

    try {
        var res = await fetch("/api/v1/documents");
        if (!res.ok) throw new Error(res.statusText);
        var data = await res.json();

        if (data.files.length === 0) {
            docList.innerHTML =
                '<div class="panel-empty">'
                + "No documents uploaded yet.</div>";
            return;
        }

        docList.innerHTML = "";
        data.files.forEach(function(f) {
            var item = document.createElement("div");
            item.className = "doc-item";

            var info = document.createElement("div");
            info.className = "doc-info";

            var name = document.createElement("div");
            name.className = "doc-name";
            name.textContent = f.filename;

            var size = document.createElement("div");
            size.className = "doc-size";
            size.textContent = f.size_kb > 0
                ? f.size_kb + " KB \u00b7 "
                    + f.chunk_count + " chunks"
                : f.chunk_count
                    + " chunks (file removed)";

            info.appendChild(name);
            info.appendChild(size);

            var btn = document.createElement("button");
            btn.className = "btn danger";
            btn.textContent = "Delete";
            btn.addEventListener("click", function() {
                deleteDocument(f.filename);
            });

            item.appendChild(info);
            item.appendChild(btn);
            docList.appendChild(item);
        });
    } catch (err) {
        docList.innerHTML =
            '<div class="panel-empty">'
            + "Failed to load documents.</div>";
    }
}

async function deleteDocument(filename) {
    if (!confirm(
        "Delete " + filename
        + "? The vector store will be re-indexed."
    )) return;

    try {
        var res = await fetch(
            "/api/v1/documents/"
            + encodeURIComponent(filename),
            { method: "DELETE" }
        );
        if (!res.ok) throw new Error(res.statusText);
        var data = await res.json();
        if (data.status === "ok") {
            loadDocuments();
            fetchStatus();
        }
    } catch (err) {
        alert("Delete failed.");
    }
}

/* ── Model Panel ──────────────────────────────────────────── */

function toggleModelPanel() {
    docPanel.classList.remove("open");
    modelPanel.classList.toggle("open");
    if (modelPanel.classList.contains("open")) {
        loadModels();
    }
}

async function loadModels() {
    modelList.innerHTML =
        '<div class="panel-empty">Loading...</div>';

    try {
        var res = await fetch("/api/v1/models");
        if (!res.ok) throw new Error(res.statusText);
        var data = await res.json();

        if (data.models.length === 0) {
            modelList.innerHTML =
                '<div class="panel-empty">'
                + "No models configured.</div>";
            return;
        }

        modelList.innerHTML = "";
        data.models.forEach(function(m) {
            var item = document.createElement("div");
            item.className = "model-item";
            item.id = "model-"
                + m.name.replace(/[:.]/g, "-");

            var info = document.createElement("div");
            info.className = "model-info";

            var name = document.createElement("div");
            name.className = "model-name";
            name.textContent = m.name;

            var meta = document.createElement("div");
            meta.className = "model-meta";

            if (m.downloaded) {
                meta.innerHTML =
                    '<span class="badge downloaded">'
                    + "Downloaded</span>";
                if (m.size_mb > 0) {
                    meta.innerHTML +=
                        " " + m.size_mb + " MB";
                }
            } else {
                meta.textContent = "Not downloaded";
            }

            info.appendChild(name);
            info.appendChild(meta);
            item.appendChild(info);

            if (m.downloaded) {
                var rmBtn =
                    document.createElement("button");
                rmBtn.className = "btn danger";
                rmBtn.textContent = "Remove";
                rmBtn.addEventListener(
                    "click",
                    function() {
                        deleteModel(m.name, rmBtn);
                    }
                );
                item.appendChild(rmBtn);
            } else {
                var btn =
                    document.createElement("button");
                btn.className = "btn";
                btn.textContent = "Pull";
                btn.addEventListener(
                    "click",
                    function() {
                        pullModel(m.name, btn);
                    }
                );
                item.appendChild(btn);
            }

            modelList.appendChild(item);
        });
    } catch (err) {
        modelList.innerHTML =
            '<div class="panel-empty">'
            + "Failed to load models.</div>";
    }
}

async function pullModel(name, btn) {
    btn.disabled = true;
    btn.textContent = "Starting...";

    try {
        var res = await fetch("/api/v1/models/pull", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ model: name }),
        });
        if (!res.ok) {
            var data = await res.json().catch(
                function() { return {}; }
            );
            throw new Error(
                data.detail || res.statusText
            );
        }

        pollPullStatus(name, btn);
    } catch (err) {
        btn.textContent = "Error";
        btn.disabled = false;
        setTimeout(function() {
            btn.textContent = "Pull";
        }, 3000);
    }
}

function pollPullStatus(name, btn) {
    var interval = setInterval(
        async function() {
            try {
                var res = await fetch(
                    "/api/v1/models/"
                    + encodeURIComponent(name)
                    + "/status"
                );
                if (!res.ok) {
                    clearInterval(interval);
                    return;
                }
                var data = await res.json();

                if (data.status === "pulling") {
                    btn.textContent =
                        data.progress || "Pulling...";
                } else if (
                    data.status === "completed"
                ) {
                    clearInterval(interval);
                    btn.textContent = "Done!";
                    btn.className =
                        "btn downloaded-btn";
                    setTimeout(function() {
                        loadModels();
                        fetchStatus();
                    }, 1000);
                } else if (
                    data.status === "error"
                ) {
                    clearInterval(interval);
                    btn.textContent = "Error";
                    btn.disabled = false;
                    setTimeout(function() {
                        btn.textContent = "Pull";
                    }, 3000);
                }
            } catch (err) {
                clearInterval(interval);
            }
        },
        2000
    );
}

async function deleteModel(name, btn) {
    if (!confirm(
        "Remove model " + name + "?"
    )) return;

    btn.disabled = true;
    btn.textContent = "Removing...";

    try {
        var res = await fetch(
            "/api/v1/models/"
            + encodeURIComponent(name),
            { method: "DELETE" }
        );
        if (!res.ok) {
            var data = await res.json().catch(
                function() { return {}; }
            );
            throw new Error(
                data.detail || res.statusText
            );
        }
        btn.textContent = "Removed!";
        setTimeout(function() {
            loadModels();
            fetchStatus();
        }, 1000);
    } catch (err) {
        btn.textContent = "Error";
        btn.disabled = false;
        setTimeout(function() {
            btn.textContent = "Remove";
        }, 3000);
    }
}

/* ── Initialization ─────────────────────────────────────────── */

migrateOldHistory();
fetchStatus();

var activeId = getActiveId();
if (activeId) {
    switchToChat(activeId);
} else {
    renderSidebar();
}
