/**
 * Main application client utilities
 */

document.addEventListener('DOMContentLoaded', () => {
    // Show spinner on video analyze form submit
    const analyzeForm = document.getElementById('analyzeForm');
    if (analyzeForm) {
        analyzeForm.addEventListener('submit', (e) => {
            const btn = document.getElementById('analyzeSubmitBtn');
            if (btn) {
                btn.disabled = true;
                btn.innerHTML = `
                    <span class="spinner-border spinner-border-sm me-2" role="status" aria-hidden="true"></span>
                    Extracting Transcript & Indexing...
                `;
            }
        });
    }

    // Flashcard UI Controller
    const flashcardElem = document.getElementById('flashcard');
    if (flashcardElem) {
        initFlashcardController();
    }
});

function initFlashcardController() {
    const card = document.getElementById('flashcard');
    const frontText = document.getElementById('cardQuestion');
    const backText = document.getElementById('cardAnswer');
    const counter = document.getElementById('cardCounter');
    const progressBar = document.getElementById('cardProgressBar');
    const prevBtn = document.getElementById('prevCardBtn');
    const nextBtn = document.getElementById('nextCardBtn');
    const flipBtn = document.getElementById('flipCardBtn');

    // Parse cards data injected in template
    const rawData = document.getElementById('flashcardsData');
    if (!rawData) return;
    
    let cards = [];
    try {
        cards = JSON.parse(rawData.textContent || '[]');
    } catch (e) {
        cards = [];
    }

    if (cards.length === 0) return;

    let currentIndex = 0;

    function renderCard(index) {
        card.classList.remove('flipped');
        const item = cards[index];
        frontText.textContent = item.question;
        backText.textContent = item.answer;
        
        counter.textContent = `Card ${index + 1} of ${cards.length}`;
        const pct = Math.round(((index + 1) / cards.length) * 100);
        if (progressBar) progressBar.style.width = `${pct}%`;

        prevBtn.disabled = index === 0;
        nextBtn.disabled = index === cards.length - 1;
    }

    card.addEventListener('click', () => {
        card.classList.toggle('flipped');
    });

    if (flipBtn) {
        flipBtn.addEventListener('click', () => {
            card.classList.toggle('flipped');
        });
    }

    if (prevBtn) {
        prevBtn.addEventListener('click', () => {
            if (currentIndex > 0) {
                currentIndex--;
                renderCard(currentIndex);
            }
        });
    }

    if (nextBtn) {
        nextBtn.addEventListener('click', () => {
            if (currentIndex < cards.length - 1) {
                currentIndex++;
                renderCard(currentIndex);
            }
        });
    }

    // Keyboard navigation (Arrow keys & Space)
    document.addEventListener('keydown', (e) => {
        if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
        if (e.code === 'Space') {
            e.preventDefault();
            card.classList.toggle('flipped');
        } else if (e.code === 'ArrowRight' && currentIndex < cards.length - 1) {
            currentIndex++;
            renderCard(currentIndex);
        } else if (e.code === 'ArrowLeft' && currentIndex > 0) {
            currentIndex--;
            renderCard(currentIndex);
        }
    });

    renderCard(0);
}

// Copy to clipboard helper
function copyToClipboard(elementId, btn) {
    const el = document.getElementById(elementId);
    if (!el) return;
    navigator.clipboard.writeText(el.innerText || el.textContent).then(() => {
        const originalText = btn.innerHTML;
        btn.innerHTML = '<i class="bi bi-check2 me-1"></i> Copied!';
        setTimeout(() => {
            btn.innerHTML = originalText;
        }, 2000);
    });
}
