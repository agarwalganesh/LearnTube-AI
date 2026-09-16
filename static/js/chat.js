/**
 * AI Video Chat Assistant Client Logic
 */

document.addEventListener('DOMContentLoaded', () => {
    const chatForm = document.getElementById('chatForm');
    const questionInput = document.getElementById('questionInput');
    const chatMessages = document.getElementById('chatMessages');
    const clearChatBtn = document.getElementById('clearChatBtn');
    const sendBtn = document.getElementById('sendBtn');

    if (!chatForm || !questionInput) return;

    const videoId = chatForm.dataset.videoId;

    // Auto-scroll to bottom of chat
    function scrollToBottom() {
        if (chatMessages) {
            chatMessages.scrollTop = chatMessages.scrollHeight;
        }
    }
    scrollToBottom();

    // Markdown/text parser for student-friendly formatting
    function formatMessage(text) {
        if (!text) return '';
        let escaped = text
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;");

        // Code blocks: ```code```
        escaped = escaped.replace(/```([\s\S]*?)```/g, '<pre class="bg-dark text-white p-2 rounded my-2"><code>$1</code></pre>');
        // Inline code: `code`
        escaped = escaped.replace(/`([^`]+)`/g, '<code class="bg-light text-danger px-1 rounded">$1</code>');
        // Bold: **text**
        escaped = escaped.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
        // Lists: - item
        escaped = escaped.replace(/(?:^|\n)- (.+)/g, '<li class="ms-3">$1</li>');
        // Newlines to <br>
        escaped = escaped.replace(/\n/g, '<br>');
        return escaped;
    }

    // Append a message bubble to the chat container
    function appendBubble(role, content) {
        const bubble = document.createElement('div');
        bubble.className = `chat-bubble ${role === 'user' ? 'chat-bubble-user' : 'chat-bubble-assistant'}`;
        
        if (role === 'user') {
            bubble.textContent = content;
        } else {
            bubble.innerHTML = formatMessage(content);
        }

        chatMessages.appendChild(bubble);
        scrollToBottom();
    }

    // Show typing animation
    function showTypingIndicator() {
        const typingElem = document.createElement('div');
        typingElem.id = 'typingIndicator';
        typingElem.className = 'chat-bubble chat-bubble-assistant d-flex align-items-center gap-1';
        typingElem.innerHTML = `
            <span class="typing-dot"></span>
            <span class="typing-dot"></span>
            <span class="typing-dot"></span>
            <small class="text-muted ms-2">AI is thinking...</small>
        `;
        chatMessages.appendChild(typingElem);
        scrollToBottom();
        return typingElem;
    }

    // Handle message submit
    async function sendMessage(question) {
        if (!question || !question.trim()) return;

        // Render user message immediately
        appendBubble('user', question);
        questionInput.value = '';
        questionInput.disabled = true;
        sendBtn.disabled = true;

        const typingIndicator = showTypingIndicator();

        try {
            const response = await fetch(`/api/chat/${videoId}`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-Requested-With': 'XMLHttpRequest'
                },
                body: JSON.stringify({ question: question })
            });

            let data;
            const contentType = response.headers.get('content-type') || '';
            if (contentType.includes('application/json')) {
                data = await response.json();
            } else {
                const text = await response.text();
                if (response.status === 504) {
                    data = { success: false, error: 'Server took too long to generate an answer (Gateway Timeout 504). Please try asking a more specific question.' };
                } else {
                    data = { success: false, error: `Server returned error (${response.status}). Please try again.` };
                }
            }

            typingIndicator.remove();

            if (response.ok && data.success) {
                appendBubble('assistant', data.answer);
            } else {
                const errorMsg = data.answer || data.error || `Error (${response.status}): Could not get an answer.`;
                appendBubble('assistant', `<span class="text-danger"><i class="bi bi-exclamation-triangle me-1"></i> ${errorMsg}</span>`);
            }
        } catch (err) {
            typingIndicator.remove();
            console.error('Chat error:', err);
            const isOffline = !navigator.onLine;
            const displayMsg = isOffline 
                ? 'Your internet connection seems to be offline. Please check your network.' 
                : `Request failed (${err.message || 'Server timeout'}). Please try again.`;
            appendBubble('assistant', `<span class="text-danger"><i class="bi bi-exclamation-circle me-1"></i> ${displayMsg}</span>`);
        } finally {
            questionInput.disabled = false;
            sendBtn.disabled = false;
            questionInput.focus();
            scrollToBottom();
        }
    }

    chatForm.addEventListener('submit', (e) => {
        e.preventDefault();
        const q = questionInput.value.trim();
        if (q) sendMessage(q);
    });

    // Preset suggestion pills
    document.querySelectorAll('.preset-query-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            const query = btn.dataset.query;
            if (query) {
                questionInput.value = query;
                sendMessage(query);
            }
        });
    });

    // Clear Chat
    if (clearChatBtn) {
        clearChatBtn.addEventListener('click', async () => {
            if (!confirm('Clear all conversation history for this video?')) return;
            try {
                const res = await fetch(`/api/chat/${videoId}/clear`, { method: 'POST' });
                if (res.ok) {
                    chatMessages.innerHTML = `
                        <div class="chat-bubble chat-bubble-assistant">
                            Hello! I am your AI study tutor for this video. Ask me anything about the content, definitions, formulas, or concepts explained in the video!
                        </div>
                    `;
                }
            } catch (e) {
                console.error('Failed to clear chat', e);
            }
        });
    }
});
