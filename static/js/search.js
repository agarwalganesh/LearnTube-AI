/**
 * Semantic Smart Search Client Logic
 */

document.addEventListener('DOMContentLoaded', () => {
    const searchForm = document.getElementById('searchForm');
    const searchInput = document.getElementById('searchInput');

    // Preset search chips
    document.querySelectorAll('.search-chip').forEach(chip => {
        chip.addEventListener('click', () => {
            const query = chip.dataset.query;
            if (searchInput && query) {
                searchInput.value = query;
                if (searchForm) {
                    searchForm.submit();
                }
            }
        });
    });
});
