/**
 * Neural Network Lottery Optimizer - Frontend Application
 * Real-time visualization with WebSocket updates
 */

// State
let socket = null;
let state = {
    poolSize: 36,
    drawSize: 5,
    matchRequired: 3,
    coverageHistory: [],
    lastUpdate: Date.now(),
    fps: 0,
    frameCount: 0,
    lastFpsUpdate: Date.now()
};

// Particle system
const particles = [];
const particleCanvas = document.getElementById('particle-canvas');
const particleCtx = particleCanvas.getContext('2d');

// Neural network canvas
const neuralCanvas = document.getElementById('neural-canvas');
const neuralCtx = neuralCanvas.getContext('2d');
let neuralPhase = 0;

// Coverage graph
const graphCanvas = document.getElementById('coverage-graph');
const graphCtx = graphCanvas.getContext('2d');

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    initializeSocket();
    initializeNumberGrid();
    resizeCanvases();
    requestAnimationFrame(animate);
    loadRecords();
});

window.addEventListener('resize', resizeCanvases);

function resizeCanvases() {
    particleCanvas.width = window.innerWidth;
    particleCanvas.height = window.innerHeight;
}

// Socket.IO Connection
function initializeSocket() {
    socket = io();

    socket.on('connect', () => {
        console.log('Connected to server');
        updateStatus('connected', 'Connected');
    });

    socket.on('disconnect', () => {
        console.log('Disconnected from server');
        updateStatus('disconnected', 'Disconnected');
    });

    socket.on('training_update', (data) => {
        handleTrainingUpdate(data);
    });
}

function updateStatus(status, text) {
    const indicator = document.getElementById('status-indicator');
    indicator.className = 'status-indicator ' + status;
    indicator.querySelector('.status-text').textContent = text;
}

// Handle training updates
function handleTrainingUpdate(data) {
    // Update configuration
    if (data.pool_size !== state.poolSize) {
        state.poolSize = data.pool_size;
        state.drawSize = data.draw_size;
        state.matchRequired = data.match_required;
        initializeNumberGrid();
    }

    // Update header
    document.getElementById('wheel-name').textContent = data.wheel_name;
    document.getElementById('pool-info').textContent = `Pool: ${data.pool_size}`;
    document.getElementById('draw-info').textContent = `Pick: ${data.draw_size}`;
    document.getElementById('match-info').textContent = `Match: ${data.match_required}+`;

    // Update stats
    document.getElementById('stat-generation').textContent = data.generation;
    document.getElementById('stat-coverage').textContent = data.coverage.toFixed(1) + '%';
    document.getElementById('stat-tickets').textContent = data.num_tickets;
    document.getElementById('stat-best').textContent = data.best_coverage.toFixed(1) + '%';
    document.getElementById('stat-efficiency').textContent = (data.efficiency || 0).toFixed(2);
    document.getElementById('stat-time').textContent = formatTime(data.elapsed_time || 0);

    // Update progress bar
    const progressFill = document.getElementById('progress-fill');
    const progressText = document.getElementById('progress-text');
    progressFill.style.width = data.coverage + '%';
    progressText.textContent = data.coverage.toFixed(1) + '%';

    // Update ghost panel
    updateGhostPanel(data);

    // Update number grid with heat map and selection
    updateNumberGrid(data.heat_map || {}, data.current_ticket || []);

    // Update tickets display
    updateTicketsList(data.tickets || []);

    // Add to coverage history
    state.coverageHistory.push(data.coverage);
    if (state.coverageHistory.length > 100) {
        state.coverageHistory.shift();
    }

    // Update status
    if (data.running) {
        updateStatus('training', 'Training...');
    } else {
        updateStatus('connected', 'Training Complete');
    }

    // Check for celebration
    if (data.is_new_record && !state.celebrated) {
        celebrate('🏆 NEW RECORD!');
        state.celebrated = true;
        emitParticles(window.innerWidth / 2, window.innerHeight / 2, 100, '#00ff80');
    }

    state.lastUpdate = Date.now();
}

function formatTime(seconds) {
    if (seconds < 60) return seconds.toFixed(0) + 's';
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}m ${secs}s`;
}

// Ghost Panel
function updateGhostPanel(data) {
    const panel = document.getElementById('ghost-panel');
    const content = document.getElementById('ghost-content');

    if (data.ghost_tickets === null) {
        content.innerHTML = `
            <p class="ghost-record">No record yet!</p>
            <p class="ghost-status new-record">🆕 Setting first record!</p>
        `;
        panel.classList.remove('new-record');
    } else {
        const ticketDiff = data.num_tickets - data.ghost_tickets;
        let statusClass = 'behind';
        let statusText = '';

        if (data.best_coverage >= data.ghost_coverage) {
            if (ticketDiff < 0) {
                statusClass = 'ahead';
                statusText = `🏆 ${-ticketDiff} FEWER TICKETS!`;
                panel.classList.add('new-record');
            } else if (ticketDiff === 0) {
                statusClass = 'ahead';
                statusText = '✅ Matched record!';
                panel.classList.add('new-record');
            } else {
                statusText = `⚠️ ${ticketDiff} more tickets`;
                panel.classList.remove('new-record');
            }
        } else {
            const coverageDiff = data.best_coverage - data.ghost_coverage;
            statusText = `📈 ${coverageDiff.toFixed(1)}% vs goal`;
            panel.classList.remove('new-record');
        }

        content.innerHTML = `
            <p class="ghost-record">Record: ${data.ghost_tickets} tickets @ ${data.ghost_coverage.toFixed(1)}%</p>
            <p class="ghost-status ${statusClass}">${statusText}</p>
        `;
    }
}

// Number Grid
function initializeNumberGrid() {
    const grid = document.getElementById('number-grid');
    grid.innerHTML = '';

    for (let i = 0; i < state.poolSize; i++) {
        const cell = document.createElement('div');
        cell.className = 'number-cell';
        cell.id = `number-${i}`;
        cell.textContent = i + 1;
        grid.appendChild(cell);
    }

    // Adjust grid columns based on pool size
    const cols = state.poolSize <= 42 ? 6 : state.poolSize <= 56 ? 7 : 8;
    grid.style.gridTemplateColumns = `repeat(${cols}, 1fr)`;
}

function updateNumberGrid(heatMap, selectedNumbers) {
    for (let i = 0; i < state.poolSize; i++) {
        const cell = document.getElementById(`number-${i}`);
        if (!cell) continue;

        const heat = heatMap[i] || 0;
        const isSelected = selectedNumbers.includes(i);

        cell.classList.remove('hot', 'selected');

        if (isSelected) {
            cell.classList.add('selected');
        } else if (heat > 0.5) {
            cell.classList.add('hot');
        }
    }
}

// Tickets List
function updateTicketsList(tickets) {
    const list = document.getElementById('ticket-list');
    const countSpan = document.getElementById('ticket-count');

    countSpan.textContent = `(${tickets.length})`;

    // Only update if tickets changed
    const currentCount = list.children.length;
    if (currentCount === tickets.length) return;

    // Clear and rebuild (could optimize with diff)
    list.innerHTML = '';

    tickets.forEach((ticket, index) => {
        const item = document.createElement('div');
        item.className = 'ticket-item';
        if (index === tickets.length - 1) {
            item.classList.add('new');
        }

        const numbers = ticket.map(n => String(n + 1).padStart(2, '0')).join('  ');
        item.innerHTML = `
            <span class="ticket-number">#${String(index + 1).padStart(3, '0')}</span>
            <span class="ticket-numbers">${numbers}</span>
        `;

        list.appendChild(item);
    });

    // Scroll to bottom
    list.scrollTop = list.scrollHeight;
}

// Load Hall of Fame records
async function loadRecords() {
    try {
        const response = await fetch('/api/records');
        const records = await response.json();
        updateHallOfFame(records);
    } catch (e) {
        console.log('Could not load records');
    }
}

function updateHallOfFame(records) {
    const list = document.getElementById('hall-list');

    if (!records || records.length === 0) {
        list.innerHTML = '<p class="no-records">No records yet!</p>';
        return;
    }

    list.innerHTML = records.map(r => `
        <div class="hall-item">
            <span class="hall-wheel">${r.wheel}</span>
            <span class="hall-tickets">${r.num_tickets} tickets</span>
        </div>
    `).join('');
}

// Celebration
function celebrate(text) {
    const div = document.createElement('div');
    div.className = 'celebration';
    div.textContent = text;
    document.body.appendChild(div);

    setTimeout(() => div.remove(), 2000);
}

// Particle System
function emitParticles(x, y, count, color) {
    for (let i = 0; i < count; i++) {
        const angle = Math.random() * Math.PI * 2;
        const speed = 100 + Math.random() * 200;
        particles.push({
            x, y,
            vx: Math.cos(angle) * speed,
            vy: Math.sin(angle) * speed - 100,
            color: color || '#00ffff',
            size: 3 + Math.random() * 5,
            life: 1 + Math.random() * 1
        });
    }
}

function updateParticles(dt) {
    for (let i = particles.length - 1; i >= 0; i--) {
        const p = particles[i];
        p.x += p.vx * dt;
        p.y += p.vy * dt;
        p.vy += 300 * dt; // gravity
        p.life -= dt;
        p.size *= 0.98;

        if (p.life <= 0 || p.size < 0.5) {
            particles.splice(i, 1);
        }
    }
}

function drawParticles() {
    particleCtx.clearRect(0, 0, particleCanvas.width, particleCanvas.height);

    particles.forEach(p => {
        particleCtx.beginPath();
        particleCtx.arc(p.x, p.y, p.size, 0, Math.PI * 2);
        particleCtx.fillStyle = p.color;
        particleCtx.globalAlpha = p.life;
        particleCtx.fill();
    });

    particleCtx.globalAlpha = 1;
}

// Neural Network Visualization
function drawNeuralNetwork() {
    const canvas = neuralCanvas;
    const ctx = neuralCtx;
    const w = canvas.width;
    const h = canvas.height;

    ctx.fillStyle = '#1e1e2e';
    ctx.fillRect(0, 0, w, h);

    neuralPhase += 0.02;

    const layers = [state.poolSize > 36 ? 8 : 6, 8, 6, state.poolSize > 36 ? 8 : 6];
    const layerSpacing = w / (layers.length + 1);

    // Draw connections first
    for (let l = 0; l < layers.length - 1; l++) {
        const x1 = layerSpacing * (l + 1);
        const x2 = layerSpacing * (l + 2);

        for (let i = 0; i < layers[l]; i++) {
            const y1 = (h / (layers[l] + 1)) * (i + 1);

            for (let j = 0; j < Math.min(3, layers[l + 1]); j++) {
                const y2 = (h / (layers[l + 1] + 1)) * (j + 1);

                const activation = 0.3 + 0.7 * Math.abs(Math.sin(neuralPhase + l * 0.5 + i * 0.3));
                ctx.strokeStyle = `rgba(180, 0, 255, ${activation * 0.3})`;
                ctx.lineWidth = 1;
                ctx.beginPath();
                ctx.moveTo(x1, y1);
                ctx.lineTo(x2, y2);
                ctx.stroke();
            }
        }
    }

    // Draw nodes
    for (let l = 0; l < layers.length; l++) {
        const x = layerSpacing * (l + 1);

        for (let i = 0; i < layers[l]; i++) {
            const y = (h / (layers[l] + 1)) * (i + 1);
            const activation = 0.3 + 0.7 * Math.abs(Math.sin(neuralPhase + l * 0.5 + i * 0.3));

            // Glow
            const gradient = ctx.createRadialGradient(x, y, 0, x, y, 15);
            gradient.addColorStop(0, `rgba(0, 255, 255, ${activation * 0.5})`);
            gradient.addColorStop(1, 'rgba(0, 255, 255, 0)');
            ctx.fillStyle = gradient;
            ctx.fillRect(x - 15, y - 15, 30, 30);

            // Node
            const radius = 4 + activation * 4;
            ctx.beginPath();
            ctx.arc(x, y, radius, 0, Math.PI * 2);

            // Color based on activation
            const hue = 180 + activation * 60; // cyan to green
            ctx.fillStyle = `hsl(${hue}, 100%, 60%)`;
            ctx.fill();
        }
    }
}

// Coverage Graph
function drawCoverageGraph() {
    const canvas = graphCanvas;
    const ctx = graphCtx;
    const w = canvas.width;
    const h = canvas.height;

    ctx.fillStyle = '#1e1e2e';
    ctx.fillRect(0, 0, w, h);

    const data = state.coverageHistory;
    if (data.length < 2) return;

    const padding = 30;
    const graphW = w - padding * 2;
    const graphH = h - padding * 2;

    // Grid lines
    ctx.strokeStyle = '#2a2a3e';
    ctx.lineWidth = 1;
    for (let i = 0; i <= 4; i++) {
        const y = padding + (graphH / 4) * i;
        ctx.beginPath();
        ctx.moveTo(padding, y);
        ctx.lineTo(w - padding, y);
        ctx.stroke();

        // Labels
        ctx.fillStyle = '#646478';
        ctx.font = '10px Rajdhani';
        ctx.fillText((100 - i * 25) + '%', 5, y + 3);
    }

    // Draw line
    ctx.strokeStyle = '#00ff80';
    ctx.lineWidth = 2;
    ctx.beginPath();

    const maxVal = 100;
    const minVal = 0;

    data.forEach((val, i) => {
        const x = padding + (i / (data.length - 1)) * graphW;
        const y = padding + (1 - (val - minVal) / (maxVal - minVal)) * graphH;

        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
    });

    ctx.stroke();

    // Glow effect
    ctx.strokeStyle = 'rgba(0, 255, 128, 0.3)';
    ctx.lineWidth = 6;
    ctx.stroke();
}

// Animation Loop
let lastTime = 0;

function animate(time) {
    const dt = (time - lastTime) / 1000;
    lastTime = time;

    // FPS counter
    state.frameCount++;
    if (time - state.lastFpsUpdate > 1000) {
        state.fps = state.frameCount;
        state.frameCount = 0;
        state.lastFpsUpdate = time;
        document.getElementById('fps-counter').textContent = state.fps + ' FPS';
    }

    // Update and draw
    updateParticles(dt);
    drawParticles();
    drawNeuralNetwork();
    drawCoverageGraph();

    requestAnimationFrame(animate);
}
