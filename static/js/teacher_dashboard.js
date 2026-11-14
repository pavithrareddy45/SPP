/**
 * ==============================================================================
 * Teacher Dashboard Interactive Logic
 * File: static/js/teacher_dashboard.js
 * Project: Student Performance Predictor
 *
 * Provides:
 * - Dynamic table filtering by Performance Tier (All, Distinction, Merit, Pass, At-Risk)
 * - Empty match detection and feedback
 * ==============================================================================
 */

document.addEventListener("DOMContentLoaded", function () {
    const filterButtons = document.querySelectorAll("#tierFilterGroup .filter-btn");
    const tableRows = document.querySelectorAll("#studentRecordsTable tbody tr.table-row");
    const noMatchRow = document.getElementById("noFilterMatchRow");

    if (!filterButtons.length || !tableRows.length) {
        return;
    }

    filterButtons.forEach(button => {
        button.addEventListener("click", function () {
            // 1. Update active button state
            filterButtons.forEach(btn => btn.classList.remove("active"));
            this.classList.add("active");

            const selectedTier = this.getAttribute("data-filter");
            let visibleCount = 0;

            // 2. Filter rows
            tableRows.forEach(row => {
                const rowTier = row.getAttribute("data-tier");

                if (selectedTier === "all" || rowTier === selectedTier) {
                    row.style.display = "";
                    visibleCount++;
                } else {
                    row.style.display = "none";
                }
            });

            // 3. Toggle empty state indicator if no records match
            if (noMatchRow) {
                if (visibleCount === 0) {
                    noMatchRow.style.display = "";
                } else {
                    noMatchRow.style.display = "none";
                }
            }
        });
    });
});
