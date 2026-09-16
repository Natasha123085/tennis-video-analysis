/* global Chart */

const placementCtx = document.getElementById("placementChart");
const shotTypeCtx = document.getElementById('shotTypeChart');
if (shotTypeCtx && window.shotTypeCounts) {
    new Chart(shotTypeCtx, {
        type: 'bar',
        data: {
            labels: ['Serve', 'Forehand', 'Backhand', 'Volley', 'Smash'],
            datasets: [{
                label: 'Shot Count',
                data: [
                    window.shotTypeCounts.serve_candidate || 0,
                    window.shotTypeCounts.forehand_like || 0,
                    window.shotTypeCounts.backhand_like || 0,
                    window.shotTypeCounts.volley_candidate || 0,
                    window.shotTypeCounts.smash_candidate || 0
                ]
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false
        }
    });
}
new Chart(placementCtx, {
    type: "bar",
    data: {
        labels: ["Left", "Middle", "Right", "Short", "Mid", "Deep"],
        datasets: [{
            label: "Shots",
            data: [
                window.shotDistribution.left,
                window.shotDistribution.middle,
                window.shotDistribution.right,
                window.shotDistribution.short,
                window.shotDistribution.mid,
                window.shotDistribution.deep
            ],
            borderRadius: 8
        }]
    },
    options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
            legend: { display: false }
        },
        scales: {
            y: {
                beginAtZero: true
            }
        }
    }
});

const directionCtx = document.getElementById("directionChart");

new Chart(directionCtx, {
    type: "doughnut",
    data: {
        labels: ["Crosscourt", "Down the Line", "Middle"],
        datasets: [{
            data: [
                window.directionDistribution.crosscourt,
                window.directionDistribution.down_the_line,
                window.directionDistribution.middle
            ]
        }]
    },
    options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
            legend: {
                position: "bottom"
            }
        }
    }
});