// Dùng URL tương đối để không phụ thuộc port server.
const API_BASE_URL = "";

const chatForm = document.getElementById("chatForm");
const questionInput = document.getElementById("questionInput");
const modeSelect = document.getElementById("modeSelect");
const chatMessages = document.getElementById("chatMessages");
const sendButton = document.getElementById("sendButton");

/**
 * Add one message bubble into chat area.
 * @param {string} text - Message content.
 * @param {"user"|"bot"} role - Who sends this message.
 * @returns {HTMLDivElement} Message element that was appended.
 */
function appendMessage(text, role) {
    const message = document.createElement("div");
    message.classList.add("message", role);
    message.textContent = text;

    chatMessages.appendChild(message);
    scrollToBottom();

    return message;
}

/**
 * Keep viewport at newest message.
 */
function scrollToBottom() {
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

/**
 * Toggle input/button while waiting for server.
 * @param {boolean} loading
 */
function setLoadingState(loading) {
    sendButton.disabled = loading;
    questionInput.disabled = loading;
}

/**
 * Call backend chat API.
 * @param {string} question
 * @returns {Promise<string>}
 */
async function askBot(question, mode) {
    const response = await fetch(`${API_BASE_URL}/chat`, {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({ question, mode })
    });

    if (!response.ok) {
        throw new Error(`API error: ${response.status}`);
    }

    const data = await response.json();
    return data;
}

chatForm.addEventListener("submit", async (event) => {
    event.preventDefault();

    const question = questionInput.value.trim();
    const mode = parseInt(modeSelect.value, 10);
    if (!question) {
        return;
    }

    appendMessage(question, "user");
    questionInput.value = "";

    setLoadingState(true);

    const typingMessage = appendMessage("Đang gõ...", "bot");
    typingMessage.classList.add("typing");

    try {
        const data = await askBot(question, mode);
        typingMessage.remove();
        const answer = data.answer || "Không nhận được phản hồi hợp lệ từ hệ thống.";
        const normalized = data.question_normalized ? ("\n\nCâu hỏi chuẩn hóa: " + data.question_normalized) : "";
        appendMessage(answer + normalized, "bot");
    } catch (error) {
        typingMessage.remove();
        appendMessage("Có lỗi khi kết nối máy chủ. Vui lòng thử lại sau.", "bot");
        console.error(error);
    } finally {
        setLoadingState(false);
        questionInput.focus();
    }
});

appendMessage(
    "Xin chào! Mình là trợ lý của HCMUTE. Bạn cần hỗ trợ thông tin gì?",
    "bot"
);
