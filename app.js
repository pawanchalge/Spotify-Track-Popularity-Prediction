/**
 * Spotify SoundLab - Client-Side Interactive Engine & ML Telemetry
 * Integrated with Stitch Design System: SoundLab Sonic Intelligence
 * Features:
 *  - 114k Spotify Catalogue Real-Time Search & Feature Ingestion
 *  - Official Spotify Embed Player Integration (Session-Aware Web Playback)
 *  - High-Fidelity Audio Preview Stream Engine with Dynamic Waveforms
 *  - Sticky SoundLab Transport Dock
 */

// ============================================================================
// 1. TAILWIND CONFIGURATION INITIALIZER
// ============================================================================
if (typeof tailwind !== 'undefined') {
  tailwind.config = {
    darkMode: "class",
    theme: {
      extend: {
        colors: {
          "primary-container": "#1db954",
          "on-primary-fixed-variant": "#005320",
          "on-primary-container": "#004118",
          "inverse-on-surface": "#313030",
          "on-tertiary-container": "#05411f",
          "tertiary-fixed-dim": "#9ad4a5",
          "tertiary-container": "#76ae82",
          "secondary-fixed": "#69ff89",
          "on-tertiary-fixed": "#00210c",
          "background": "#131313",
          "outline-variant": "#3d4a3d",
          "on-secondary-container": "#004f1d",
          "error": "#ffb4ab",
          "on-background": "#e5e2e1",
          "tertiary": "#9ad4a5",
          "surface-container-lowest": "#0e0e0e",
          "on-secondary-fixed-variant": "#00531f",
          "tertiary-fixed": "#b5f1c0",
          "secondary-container": "#00cc57",
          "surface-container-high": "#2a2a2a",
          "on-secondary-fixed": "#002108",
          "on-surface-variant": "#bccbb9",
          "outline": "#869585",
          "on-error-container": "#ffdad6",
          "surface-variant": "#353534",
          "inverse-surface": "#e5e2e1",
          "on-primary-fixed": "#002108",
          "error-container": "#93000a",
          "on-secondary": "#003913",
          "on-error": "#690005",
          "surface": "#131313",
          "primary-fixed": "#72fe8f",
          "surface-container-low": "#1c1b1b",
          "surface-tint": "#53e076",
          "on-tertiary": "#003919",
          "surface-dim": "#131313",
          "secondary": "#3de96f",
          "surface-container-highest": "#353534",
          "primary": "#53e076",
          "inverse-primary": "#006e2d",
          "on-surface": "#e5e2e1",
          "surface-container": "#201f1f",
          "surface-bright": "#393939",
          "on-tertiary-fixed-variant": "#19512d",
          "on-primary": "#003914",
          "secondary-fixed-dim": "#34e36a",
          "primary-fixed-dim": "#53e076"
        },
        borderRadius: {
          "DEFAULT": "0.25rem",
          "lg": "0.5rem",
          "xl": "0.75rem",
          "full": "9999px"
        },
        spacing: {
          "space-2xs": "0.125rem",
          "space-xl": "2rem",
          "gutter-sm": "0.75rem",
          "space-sm": "0.5rem",
          "gutter-lg": "1.75rem",
          "margin-desktop": "2.5rem",
          "margin": "2rem",
          "space-xs": "0.25rem",
          "space-md": "1rem",
          "space-3xl": "4rem",
          "gutter": "1.25rem",
          "space-lg": "1.5rem",
          "space-2xl": "3rem",
          "margin-mobile": "1rem"
        },
        fontFamily: {
          "telemetry-metric": ["JetBrains Mono", "monospace"],
          "body-sm": ["Plus Jakarta Sans", "sans-serif"],
          "display-mobile": ["Plus Jakarta Sans", "sans-serif"],
          "headline-sm": ["Plus Jakarta Sans", "sans-serif"],
          "headline-lg-mobile": ["Plus Jakarta Sans", "sans-serif"],
          "caption": ["Plus Jakarta Sans", "sans-serif"],
          "body-md": ["Plus Jakarta Sans", "sans-serif"],
          "headline-lg": ["Plus Jakarta Sans", "sans-serif"],
          "telemetry-label": ["JetBrains Mono", "monospace"],
          "display": ["Plus Jakarta Sans", "sans-serif"],
          "body-lg": ["Plus Jakarta Sans", "sans-serif"],
          "headline-md": ["Plus Jakarta Sans", "sans-serif"],
          "title-md": ["Plus Jakarta Sans", "sans-serif"]
        }
      }
    }
  };
}

// ============================================================================
// 2. ARCHETYPE PRESETS DATA
// ============================================================================
const PRESETS = {
  pop: {
    track_id: "7qiZfU4dY1lWllzX7mPBI3", // Shape of You
    name: "Mainstream Pop Hit",
    artist: "Global Chart Benchmark",
    danceability: 0.82,
    energy: 0.78,
    loudness: -5.0,
    acousticness: 0.12,
    instrumentalness: 0.00,
    valence: 0.75,
    tempo: 124,
    speechiness: 0.06,
    genre: 'pop',
    key: '1',
    explicit: false
  },
  indie: {
    track_id: "4qPNDBW1i3p13qLCt0Ki3A", // Ghost - Acoustic
    name: "Indie Folk Ballad",
    artist: "Acoustic Warmth",
    danceability: 0.44,
    energy: 0.38,
    loudness: -11.5,
    acousticness: 0.84,
    instrumentalness: 0.08,
    valence: 0.35,
    tempo: 98,
    speechiness: 0.04,
    genre: 'indie',
    key: '7',
    explicit: false
  },
  edm: {
    track_id: "6lfxq3CG4xtTiEg7opyCyx", // Festival Anthem
    name: "Festival EDM Anthem",
    artist: "Mainstage Peak",
    danceability: 0.76,
    energy: 0.94,
    loudness: -3.8,
    acousticness: 0.03,
    instrumentalness: 0.52,
    valence: 0.65,
    tempo: 128,
    speechiness: 0.07,
    genre: 'dance',
    key: '0',
    explicit: false
  },
  jazz: {
    track_id: "1EzrEOXmMH3G43AXT1y7pA", // Smooth Acoustic
    name: "Smooth Jazz Topline",
    artist: "Acoustic Dynamics",
    danceability: 0.54,
    energy: 0.32,
    loudness: -14.0,
    acousticness: 0.79,
    instrumentalness: 0.82,
    valence: 0.52,
    tempo: 110,
    speechiness: 0.05,
    genre: 'acoustic',
    key: '2',
    explicit: false
  },
  trap: {
    track_id: "5vjLSffimiIP26QG5WcN2K", // Modern Trap
    name: "Modern 808 Trap",
    artist: "Urban Dynamics",
    danceability: 0.88,
    energy: 0.74,
    loudness: -6.2,
    acousticness: 0.08,
    instrumentalness: 0.00,
    valence: 0.55,
    tempo: 140,
    speechiness: 0.28,
    genre: 'hip-hop',
    key: '9',
    explicit: true
  }
};

const PRESET_KEYS = ['pop', 'indie', 'edm', 'jazz', 'trap'];
let currentPresetIndex = 0;

// ============================================================================
// 3. MAIN APPLICATION CONTROLLER
// ============================================================================
document.addEventListener('DOMContentLoaded', () => {
  // Sliders map
  const sliders = {
    danceability: document.getElementById('slider-danceability'),
    energy: document.getElementById('slider-energy'),
    loudness: document.getElementById('slider-loudness'),
    acousticness: document.getElementById('slider-acousticness'),
    instrumentalness: document.getElementById('slider-instrumentalness'),
    valence: document.getElementById('slider-valence'),
    tempo: document.getElementById('slider-tempo'),
    speechiness: document.getElementById('slider-speechiness')
  };

  // Badges map
  const badges = {
    danceability: document.getElementById('badge-danceability'),
    energy: document.getElementById('badge-energy'),
    loudness: document.getElementById('badge-loudness'),
    acousticness: document.getElementById('badge-acousticness'),
    instrumentalness: document.getElementById('badge-instrumentalness'),
    valence: document.getElementById('badge-valence'),
    tempo: document.getElementById('badge-tempo'),
    speechiness: document.getElementById('badge-speechiness')
  };

  const toggleExplicit = document.getElementById('toggle-explicit');
  const badgeExplicit = document.getElementById('badge-explicit');
  const selectGenre = document.getElementById('select-genre');
  const selectKey = document.getElementById('select-key');
  const keyModeBadge = document.getElementById('keyModeBadge');

  // Display elements
  const scoreNumber = document.getElementById('popularityScoreNumber');
  const gaugeCircle = document.getElementById('gaugeProgressCircle');
  const verdictBadge = document.getElementById('verdictBadge');
  const streamEstMetric = document.getElementById('streamEstMetric');
  const streamCompMetric = document.getElementById('streamCompMetric');
  const playlistMatchMetric = document.getElementById('playlistMatchMetric');
  const skipRateMetric = document.getElementById('skipRateMetric');
  const radarPolygon = document.getElementById('radarPolygon');
  const waveformContainer = document.getElementById('waveformContainer');

  // Search Elements
  const searchInput = document.getElementById('searchInput');
  const clearSearchBtn = document.getElementById('clearSearchBtn');
  const searchResultsDropdown = document.getElementById('searchResultsDropdown');
  const searchResultsList = document.getElementById('searchResultsList');
  const searchStatusText = document.getElementById('searchStatusText');
  const searchResultCount = document.getElementById('searchResultCount');

  // Loaded Track Banner Elements
  const loadedTrackBanner = document.getElementById('loadedTrackBanner');
  const loadedTrackTitle = document.getElementById('loadedTrackTitle');
  const loadedTrackArtist = document.getElementById('loadedTrackArtist');
  const loadedTrackAlbum = document.getElementById('loadedTrackAlbum');
  const loadedTrackGenre = document.getElementById('loadedTrackGenre');
  const loadedActualScore = document.getElementById('loadedActualScore');
  const loadedPredictedScore = document.getElementById('loadedPredictedScore');
  const loadedAccuracyBadge = document.getElementById('loadedAccuracyBadge');
  const clearLoadedTrackBtn = document.getElementById('clearLoadedTrackBtn');
  const bannerPlayBtn = document.getElementById('bannerPlayBtn');
  const bannerSpotifyLink = document.getElementById('bannerSpotifyLink');

  // Player & Visualizer Elements
  const audioPlayToggle = document.getElementById('audioPlayToggle');
  const playerTrackTitle = document.getElementById('playerTrackTitle');
  const playerArtistName = document.getElementById('playerArtistName');
  const audioTimeDisplay = document.getElementById('audioTimeDisplay');
  const audioProgressBar = document.getElementById('audioProgressBar');
  const audioProgressFill = document.getElementById('audioProgressFill');
  const audioVolumeSlider = document.getElementById('audioVolumeSlider');
  const tabWaveformBtn = document.getElementById('tabWaveformBtn');
  const tabSpotifyEmbedBtn = document.getElementById('tabSpotifyEmbedBtn');
  const waveformView = document.getElementById('waveformView');
  const spotifyEmbedView = document.getElementById('spotifyEmbedView');
  const spotifyPlayerIframe = document.getElementById('spotifyPlayerIframe');
  const spotifyDirectLink = document.getElementById('spotifyDirectLink');

  // Sticky Dock Elements
  const spotifyDock = document.getElementById('spotifyDock');
  const dockArtwork = document.getElementById('dockArtwork');
  const dockTrackTitle = document.getElementById('dockTrackTitle');
  const dockArtistName = document.getElementById('dockArtistName');
  const dockPlayBtn = document.getElementById('dockPlayBtn');
  const dockPrevBtn = document.getElementById('dockPrevBtn');
  const dockNextBtn = document.getElementById('dockNextBtn');
  const dockProgressBar = document.getElementById('dockProgressBar');
  const dockProgressFill = document.getElementById('dockProgressFill');
  const dockCurrentTime = document.getElementById('dockCurrentTime');
  const dockTotalTime = document.getElementById('dockTotalTime');
  const dockSpotifyLink = document.getElementById('dockSpotifyLink');
  const dockCloseBtn = document.getElementById('dockCloseBtn');

  // Active track state
  let currentLoadedTrack = null;
  let activeTrackId = "7qiZfU4dY1lWllzX7mPBI3"; // Default: Shape of You

  // ============================================================================
  // 4. REAL AUDIO STREAMING & WAVEFORM ENGINE
  // ============================================================================
  const audio = new Audio();
  audio.crossOrigin = "anonymous";
  audio.volume = 0.8;
  let isAudioPlaying = false;
  let waveAnimFrame = null;

  function formatTime(seconds) {
    if (isNaN(seconds) || seconds < 0) return "0:00";
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs < 10 ? '0' : ''}${secs}`;
  }

  function setPlayingState(playing) {
    isAudioPlaying = playing;
    const playIcon = playing ? 'pause' : 'play_arrow';
    
    if (audioPlayToggle) {
      const icon = audioPlayToggle.querySelector('span');
      if (icon) icon.textContent = playIcon;
    }
    if (dockPlayBtn) {
      const icon = dockPlayBtn.querySelector('span');
      if (icon) icon.textContent = playIcon;
    }
    if (bannerPlayBtn) {
      const icon = bannerPlayBtn.querySelector('span');
      if (icon) icon.textContent = playIcon;
      const label = bannerPlayBtn.querySelector('span:last-child');
      if (label) label.textContent = playing ? 'Pause Audio' : 'Play Audio';
    }

    if (playing) {
      startWaveformAnimation();
    } else {
      stopWaveformAnimation();
    }
  }

  function toggleAudioPlayback() {
    if (!audio.src) {
      // If no src, fetch preview for active track
      if (currentLoadedTrack) {
        fetchTrackPreviewAndPlay(currentLoadedTrack);
      } else {
        // Fallback default sample audio
        audio.src = "https://audio-ssl.itunes.apple.com/itunes-assets/AudioPreview221/v4/44/c7/4f/44c74f0d-72dc-6143-d4d0-ba14d661ca0d/mzaf_9566898362556366703.plus.aac.p.m4a";
        audio.play().then(() => setPlayingState(true)).catch(e => console.log('Audio error:', e));
      }
      return;
    }

    if (audio.paused) {
      audio.play().then(() => setPlayingState(true)).catch(e => console.log('Playback error:', e));
    } else {
      audio.pause();
      setPlayingState(false);
    }
  }

  // Audio Events
  audio.addEventListener('play', () => setPlayingState(true));
  audio.addEventListener('pause', () => setPlayingState(false));
  audio.addEventListener('ended', () => {
    setPlayingState(false);
    audio.currentTime = 0;
  });

  audio.addEventListener('timeupdate', () => {
    const cur = audio.currentTime || 0;
    const dur = audio.duration || 30;
    const pct = Math.min(100, (cur / dur) * 100);

    if (audioProgressFill) audioProgressFill.style.width = `${pct}%`;
    if (dockProgressFill) dockProgressFill.style.width = `${pct}%`;
    if (audioTimeDisplay) audioTimeDisplay.textContent = `${formatTime(cur)} / ${formatTime(dur)}`;
    if (dockCurrentTime) dockCurrentTime.textContent = formatTime(cur);
    if (dockTotalTime) dockTotalTime.textContent = formatTime(dur);
  });

  // Seek bar clicks
  function handleProgressSeek(e, barEl) {
    if (!barEl || !audio.duration) return;
    const rect = barEl.getBoundingClientRect();
    const clickX = e.clientX - rect.left;
    const pct = Math.max(0, Math.min(1, clickX / rect.width));
    audio.currentTime = pct * audio.duration;
  }

  if (audioProgressBar) {
    audioProgressBar.addEventListener('click', (e) => handleProgressSeek(e, audioProgressBar));
  }
  if (dockProgressBar) {
    dockProgressBar.addEventListener('click', (e) => handleProgressSeek(e, dockProgressBar));
  }

  // Volume control
  if (audioVolumeSlider) {
    audioVolumeSlider.addEventListener('input', (e) => {
      audio.volume = parseFloat(e.target.value);
    });
  }

  // Waveform Visualizer Elements
  function generateWaveform() {
    if (!waveformContainer) return;
    waveformContainer.innerHTML = '';
    const barCount = 42;
    for (let i = 0; i < barCount; i++) {
      const bar = document.createElement('div');
      bar.className = 'w-1 rounded-full transition-all duration-150';
      const height = Math.floor(Math.random() * 26 + 6);
      bar.style.height = `${height}px`;
      if (i % 3 === 0) bar.style.backgroundColor = '#53e076';
      else if (i % 4 === 0) bar.style.backgroundColor = '#3de96f';
      else bar.style.backgroundColor = 'rgba(134, 149, 133, 0.4)';
      waveformContainer.appendChild(bar);
    }
  }
  generateWaveform();

  function startWaveformAnimation() {
    function animate() {
      if (!isAudioPlaying) return;
      const bars = waveformContainer ? waveformContainer.children : [];
      for (let i = 0; i < bars.length; i++) {
        const height = Math.floor(Math.random() * 32 + 6);
        bars[i].style.height = `${height}px`;
      }
      waveAnimFrame = setTimeout(() => {
        requestAnimationFrame(animate);
      }, 90);
    }
    animate();
  }

  function stopWaveformAnimation() {
    clearTimeout(waveAnimFrame);
    const bars = waveformContainer ? waveformContainer.children : [];
    for (let i = 0; i < bars.length; i++) {
      bars[i].style.height = `${Math.floor(Math.random() * 12 + 6)}px`;
    }
  }

  // Player View Mode Tabs
  if (tabWaveformBtn && tabSpotifyEmbedBtn) {
    tabWaveformBtn.addEventListener('click', () => {
      tabWaveformBtn.className = "px-3 py-1 rounded-full text-caption font-semibold bg-surface-container-high text-primary border border-primary/40 flex items-center gap-1.5 transition-all";
      tabSpotifyEmbedBtn.className = "px-3 py-1 rounded-full text-caption font-normal text-on-surface-variant hover:text-white bg-transparent hover:bg-surface-container border border-outline-variant/30 flex items-center gap-1.5 transition-all";
      if (waveformView) waveformView.classList.remove('hidden');
      if (spotifyEmbedView) spotifyEmbedView.classList.add('hidden');
    });

    tabSpotifyEmbedBtn.addEventListener('click', () => {
      tabSpotifyEmbedBtn.className = "px-3 py-1 rounded-full text-caption font-semibold bg-surface-container-high text-primary border border-primary/40 flex items-center gap-1.5 transition-all";
      tabWaveformBtn.className = "px-3 py-1 rounded-full text-caption font-normal text-on-surface-variant hover:text-white bg-transparent hover:bg-surface-container border border-outline-variant/30 flex items-center gap-1.5 transition-all";
      if (spotifyEmbedView) spotifyEmbedView.classList.remove('hidden');
      if (waveformView) waveformView.classList.add('hidden');
    });
  }

  // Play button listeners
  if (audioPlayToggle) audioPlayToggle.addEventListener('click', toggleAudioPlayback);
  if (dockPlayBtn) dockPlayBtn.addEventListener('click', toggleAudioPlayback);
  if (bannerPlayBtn) bannerPlayBtn.addEventListener('click', toggleAudioPlayback);

  // Dock Collapse button
  if (dockCloseBtn && spotifyDock) {
    dockCloseBtn.addEventListener('click', () => {
      spotifyDock.classList.toggle('collapsed');
      const icon = dockCloseBtn.querySelector('span');
      if (icon) {
        icon.textContent = spotifyDock.classList.contains('collapsed') ? 'keyboard_arrow_up' : 'keyboard_arrow_down';
      }
    });
  }

  // ============================================================================
  // ============================================================================
  // 5. ML SCORING ALGORITHM & REAL-TIME XGBOOST PIPELINE INFERENCE (/api/predict)
  // ============================================================================
  let predictDebounceTimer = null;
  let lastPredictionData = null;

  function calculateScoreFallback() {
    if (!sliders.danceability) return 50;

    const d = parseFloat(sliders.danceability.value);
    const e = parseFloat(sliders.energy.value);
    const l = parseFloat(sliders.loudness.value);
    const a = parseFloat(sliders.acousticness.value);
    const inst = parseFloat(sliders.instrumentalness.value);
    const v = parseFloat(sliders.valence.value);
    const t = parseFloat(sliders.tempo.value);
    const sp = parseFloat(sliders.speechiness.value);
    const explicit = toggleExplicit ? toggleExplicit.checked : false;
    const genre = selectGenre ? selectGenre.value : 'pop';

    let base = 52.0;
    base += (d - 0.5) * 22.0;
    base += (e - 0.5) * 18.0;
    const normLoud = (l + 60) / 60;
    base += (normLoud - 0.7) * 24.0;
    base -= a * 12.0;
    base -= inst * 14.0;
    base += (v - 0.5) * 10.0;

    if (t >= 115 && t <= 135) base += 6.0;
    else if (t < 85 || t > 165) base -= 4.0;

    if (genre === 'pop' || genre === 'dance' || genre === 'latin') base += 8.0;
    if (genre === 'hip-hop') base += 6.0;
    if (!explicit) base += 2.0;

    return Math.round(Math.max(8, Math.min(97, base)));
  }

  async function fetchModelPrediction() {
    if (!sliders.danceability) return;

    const payload = {
      track_genre: selectGenre ? selectGenre.value : 'pop',
      key: selectKey ? parseInt(selectKey.value) : 1,
      mode: currentLoadedTrack ? (currentLoadedTrack.mode ?? 1) : 1,
      duration_ms: currentLoadedTrack ? (currentLoadedTrack.duration_ms ?? 200000) : 200000,
      danceability: parseFloat(sliders.danceability.value),
      energy: parseFloat(sliders.energy.value),
      loudness: parseFloat(sliders.loudness.value),
      speechiness: parseFloat(sliders.speechiness.value),
      acousticness: parseFloat(sliders.acousticness.value),
      instrumentalness: parseFloat(sliders.instrumentalness.value),
      liveness: currentLoadedTrack ? (currentLoadedTrack.liveness ?? 0.12) : 0.12,
      valence: parseFloat(sliders.valence.value),
      tempo: parseFloat(sliders.tempo.value),
      time_signature: currentLoadedTrack ? (currentLoadedTrack.time_signature ?? 4) : 4,
      explicit: toggleExplicit && toggleExplicit.checked ? 1 : 0
    };

    try {
      const resp = await fetch('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: json_stringify(payload)
      });
      if (resp.ok) {
        const data = await resp.json();
        lastPredictionData = data;
        applyPredictionResults(data);
      } else {
        applyFallbackResults();
      }
    } catch (e) {
      applyFallbackResults();
    }
  }

  function json_stringify(obj) {
    return JSON.stringify(obj);
  }

  function applyPredictionResults(data) {
    const score = Math.round(data.predicted_popularity);
    if (scoreNumber) scoreNumber.textContent = score;

    if (gaugeCircle) {
      const circumference = 314.15;
      const offset = circumference - (score / 100) * circumference;
      gaugeCircle.style.strokeDashoffset = offset;
    }

    // 95% Confidence Interval
    const ciText = document.getElementById('confidenceIntervalText');
    if (ciText && data.confidence_interval_95) {
      ciText.textContent = `${data.confidence_interval_95[0]} - ${data.confidence_interval_95[1]} pts`;
    }

    // Feature Completeness Badge
    const compBadge = document.getElementById('completenessBadge');
    const compText = document.getElementById('completenessText');
    if (data.feature_completeness) {
      const count = data.feature_completeness.verified_count;
      const total = data.feature_completeness.total_count;
      if (compText) compText.textContent = `${count}/${total} Features Validated (${data.feature_completeness.percentage}%)`;
      if (compBadge) {
        if (data.feature_completeness.percentage >= 90) {
          compBadge.className = 'completeness-badge completeness-full';
        } else {
          compBadge.className = 'completeness-badge completeness-partial';
        }
      }
    }

    // Tier Verdict Badge
    if (verdictBadge && data.tier) {
      verdictBadge.innerHTML = `<span class="material-symbols-outlined text-[16px]">auto_awesome</span><span>${data.tier.title}</span>`;
      verdictBadge.style.borderColor = data.tier.color;
      verdictBadge.style.color = data.tier.color;
    }

    // Listening Behavior & Skip Retention
    if (data.listening_behavior) {
      const lb = data.listening_behavior;
      if (skipRateMetric) skipRateMetric.textContent = `${lb.skip_probability_30s}%`;
      if (completionRateMetric) completionRateMetric.textContent = `${lb.completion_rate_est}%`;

      const retSkipVal = document.getElementById('retentionSkipVal');
      const retCompVal = document.getElementById('retentionCompletionVal');
      if (retSkipVal) retSkipVal.textContent = `${lb.skip_probability_30s}%`;
      if (retCompVal) retCompVal.textContent = `${lb.completion_rate_est}%`;

      // Retention timeline bars
      const p10 = Math.min(99, Math.round(100 - (lb.skip_probability_30s * 0.35)));
      const p30 = Math.round(100 - lb.skip_probability_30s);
      const p60 = Math.round(lb.completion_rate_est * 0.95);
      const pFull = Math.round(lb.completion_rate_est * 0.85);

      const b10 = document.getElementById('bar10s');
      const b30 = document.getElementById('bar30s');
      const b60 = document.getElementById('bar60s');
      const bFull = document.getElementById('barFull');
      const t10 = document.getElementById('retention10s');
      const t30 = document.getElementById('retention30s');
      const t60 = document.getElementById('retention60s');
      const tFull = document.getElementById('retentionFull');

      if (b10 && t10) { b10.style.width = `${p10}%`; t10.textContent = `${p10}%`; }
      if (b30 && t30) { b30.style.width = `${p30}%`; t30.textContent = `${p30}%`; }
      if (b60 && t60) { b60.style.width = `${p60}%`; t60.textContent = `${p60}%`; }
      if (bFull && tFull) { bFull.style.width = `${pFull}%`; tFull.textContent = `${pFull}%`; }

      // Context Affinities
      if (lb.context_affinities) {
        const ca = lb.context_affinities;
        setContextMeter('ctxWorkoutVal', 'ctxWorkoutBar', ca.workout);
        setContextMeter('ctxPartyVal', 'ctxPartyBar', ca.party);
        setContextMeter('ctxFocusVal', 'ctxFocusBar', ca.focus_study);
        setContextMeter('ctxChillVal', 'ctxChillBar', ca.late_night_chill);
        setContextMeter('ctxCommuteVal', 'ctxCommuteBar', ca.highway_commute);
      }
    }

    // Ingested Track comparison delta
    if (currentLoadedTrack && loadedTrackBanner && !loadedTrackBanner.classList.contains('hidden')) {
      const predicted = score;
      const actual = currentLoadedTrack.popularity;
      const delta = predicted - actual;
      const accuracy = Math.max(0, 100 - Math.abs(delta));
      if (loadedPredictedScore) loadedPredictedScore.textContent = predicted;
      if (loadedAccuracyBadge) {
        const deltaSign = delta > 0 ? `+${delta}` : `${delta}`;
        loadedAccuracyBadge.textContent = `Δ ${deltaSign} pts (${accuracy.toFixed(1)}% match)`;
      }
    }
  }

  function setContextMeter(valId, barId, val) {
    const valEl = document.getElementById(valId);
    const barEl = document.getElementById(barId);
    const rounded = Math.round(val || 0);
    if (valEl) valEl.textContent = `${rounded}%`;
    if (barEl) barEl.style.width = `${rounded}%`;
  }

  function applyFallbackResults() {
    const score = calculateScoreFallback();
    if (scoreNumber) scoreNumber.textContent = score;
    if (gaugeCircle) {
      const circumference = 314.15;
      const offset = circumference - (score / 100) * circumference;
      gaugeCircle.style.strokeDashoffset = offset;
    }
  }

  // ============================================================================
  // 6. UI SYNCHRONIZATION
  // ============================================================================
  function updateUI() {
    if (!scoreNumber) return;

    // 1. Debounced async prediction call to live ML model pipeline
    clearTimeout(predictDebounceTimer);
    predictDebounceTimer = setTimeout(() => {
      fetchModelPrediction();
    }, 60);

    // 2. Sonic Radar SVG points calculation
    if (radarPolygon && sliders.danceability) {
      const d = parseFloat(sliders.danceability.value);
      const e = parseFloat(sliders.energy.value);
      const l = (parseFloat(sliders.loudness.value) + 60) / 60;
      const t = (parseFloat(sliders.tempo.value) - 60) / 160;
      const a = parseFloat(sliders.acousticness.value);
      const v = parseFloat(sliders.valence.value);

      const center = 100;
      const maxR = 80;

      function getPt(angleDeg, val) {
        const rad = (angleDeg - 90) * (Math.PI / 180);
        const r = Math.max(15, val * maxR);
        return `${(center + r * Math.cos(rad)).toFixed(1)},${(center + r * Math.sin(rad)).toFixed(1)}`;
      }

      const p1 = getPt(0, d);
      const p2 = getPt(60, e);
      const p3 = getPt(120, l);
      const p4 = getPt(180, t);
      const p5 = getPt(240, a);
      const p6 = getPt(300, v);

      radarPolygon.setAttribute('points', `${p1} ${p2} ${p3} ${p4} ${p5} ${p6}`);
    }

    // 3. Slider Badges formatting
    if (badges.danceability) {
      const d = parseFloat(sliders.danceability.value);
      const e = parseFloat(sliders.energy.value);
      const l = (parseFloat(sliders.loudness.value) + 60) / 60;
      const a = parseFloat(sliders.acousticness.value);
      const v = parseFloat(sliders.valence.value);

      badges.danceability.textContent = d.toFixed(2);
      badges.energy.textContent = e.toFixed(2);
      badges.loudness.textContent = `${parseFloat(sliders.loudness.value).toFixed(1)} dB`;
      badges.acousticness.textContent = a.toFixed(2);
      badges.instrumentalness.textContent = parseFloat(sliders.instrumentalness.value).toFixed(2);
      badges.valence.textContent = v.toFixed(2);
      badges.tempo.textContent = `${sliders.tempo.value} BPM`;
      badges.speechiness.textContent = parseFloat(sliders.speechiness.value).toFixed(2);
      if (keyModeBadge && selectGenre && selectKey) {
        keyModeBadge.textContent = `${selectGenre.options[selectGenre.selectedIndex].text} • ${selectKey.options[selectKey.selectedIndex].text}`;
      }
    }

    if (badgeExplicit && toggleExplicit) {
      badgeExplicit.textContent = toggleExplicit.checked ? 'FLAGGED: ON' : 'CLEAN (OFF)';
      badgeExplicit.className = toggleExplicit.checked
        ? 'text-telemetry-label text-error bg-surface-container-lowest border border-error/30 px-2 py-0.5 rounded text-[11px]'
        : 'text-telemetry-label text-on-surface-variant bg-surface-container-lowest border border-outline-variant/30 px-2 py-0.5 rounded text-[11px]';
    }
  }

  // ============================================================================
  // 7. SPOTIFY CATALOGUE SEARCH ENGINE
  // ============================================================================
  let localCatalog = [];
  let selectedIndex = -1;
  let currentResults = [];
  let debounceTimer = null;

  fetch('catalog.json')
    .then(res => res.json())
    .then(data => {
      localCatalog = data;
      console.log(`[SoundLab] Cached ${localCatalog.length.toLocaleString()} Spotify tracks for offline search.`);
    })
    .catch(err => {
      console.log('[SoundLab] Note: catalog.json cache fetch deferred:', err);
    });

  const TRENDING_SUGGESTIONS = [
    { track_id: "7qiZfU4dY1lWllzX7mPBI3", name: "Shape of You", artist: "Ed Sheeran", album: "÷ (Divide)", genre: "pop", popularity: 87, danceability: 0.825, energy: 0.652, loudness: -3.183, acousticness: 0.581, instrumentalness: 0.0, valence: 0.931, tempo: 95.977, speechiness: 0.0802, explicit: false, key: 1 },
    { track_id: "0VjIjW4GlUZAMYd2vXMi3b", name: "Blinding Lights", artist: "The Weeknd", album: "After Hours", genre: "pop", popularity: 90, danceability: 0.514, energy: 0.730, loudness: -5.934, acousticness: 0.001, instrumentalness: 0.0, valence: 0.334, tempo: 171.005, speechiness: 0.0598, explicit: false, key: 1 },
    { track_id: "4Dvkj6JhhA12EX05QKi792", name: "As It Was", artist: "Harry Styles", album: "Harry's House", genre: "pop", popularity: 89, danceability: 0.520, energy: 0.731, loudness: -5.338, acousticness: 0.342, instrumentalness: 0.001, valence: 0.662, tempo: 173.93, speechiness: 0.0557, explicit: false, key: 6 },
    { track_id: "3AJwUDP919kvQ9QcozQPxg", name: "Yellow", artist: "Coldplay", album: "Parachutes", genre: "rock", popularity: 88, danceability: 0.429, energy: 0.661, loudness: -7.227, acousticness: 0.002, instrumentalness: 0.0001, valence: 0.285, tempo: 173.372, speechiness: 0.0281, explicit: false, key: 11 },
    { track_id: "5PjdY0CKGZdErtk25b9Lug", name: "Stay", artist: "The Kid LAROI, Justin Bieber", album: "F*CK LOVE 3", genre: "pop", popularity: 85, danceability: 0.591, energy: 0.764, loudness: -5.484, acousticness: 0.0383, instrumentalness: 0.0, valence: 0.478, tempo: 169.928, speechiness: 0.0483, explicit: true, key: 1 },
    { track_id: "2Fxmhks0bxGSBdJ92vM42m", name: "Bad Guy", artist: "Billie Eilish", album: "WHEN WE ALL FALL ASLEEP", genre: "pop", popularity: 84, danceability: 0.701, energy: 0.425, loudness: -10.965, acousticness: 0.328, instrumentalness: 0.13, valence: 0.562, tempo: 135.128, speechiness: 0.375, explicit: false, key: 7 }
  ];

  function escapeHtml(str) {
    if (!str) return '';
    return String(str)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;');
  }

  function highlightMatch(text, query) {
    if (!query || !text) return escapeHtml(text || '');
    const cleanQuery = query.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    const regex = new RegExp(`(${cleanQuery})`, 'gi');
    return escapeHtml(text).replace(regex, '<span class="search-highlight">$1</span>');
  }

  function setSelectedIndex(idx) {
    if (!searchResultsList) return;
    const items = searchResultsList.querySelectorAll('.search-result-item');
    items.forEach((item, i) => {
      if (i === idx) {
        item.classList.add('selected');
        item.scrollIntoView({ block: 'nearest' });
      } else {
        item.classList.remove('selected');
      }
    });
    selectedIndex = idx;
  }

  function renderResults(tracks, query) {
    if (!searchResultsList || !searchResultsDropdown) return;
    searchResultsList.innerHTML = '';
    selectedIndex = -1;

    if (!tracks || tracks.length === 0) {
      searchResultsDropdown.classList.remove('hidden');
      if (searchStatusText) searchStatusText.textContent = 'NO CATALOGUE MATCHES';
      if (searchResultCount) searchResultCount.textContent = '0 found';
      searchResultsList.innerHTML = `
        <div class="p-6 text-center text-on-surface-variant">
          <span class="material-symbols-outlined text-[32px] text-outline opacity-60">search_off</span>
          <p class="text-body-sm font-semibold mt-1">No songs found for "${escapeHtml(query)}"</p>
          <p class="text-[11px] text-outline mt-0.5">Try searching artist names like "Ed Sheeran", "The Weeknd", "Coldplay", or genres like "pop", "rock"</p>
        </div>
      `;
      return;
    }

    searchResultsDropdown.classList.remove('hidden');
    if (searchStatusText) searchStatusText.textContent = query ? 'SPOTIFY CATALOGUE MATCHES' : '🔥 TRENDING CATALOGUE PICKS';
    if (searchResultCount) searchResultCount.textContent = `${tracks.length} tracks`;

    tracks.forEach((track, idx) => {
      const item = document.createElement('div');
      item.className = 'search-result-item px-3 py-2 cursor-pointer flex items-center justify-between gap-3 text-left transition-all';
      item.setAttribute('data-index', idx);

      const popScore = track.popularity !== undefined ? track.popularity : 50;
      let popBadgeClass = 'bg-surface-container-high text-on-surface-variant border-outline-variant/40';
      let popIcon = 'radio';
      if (popScore >= 80) {
        popBadgeClass = 'bg-primary/20 text-primary border-primary/50';
        popIcon = 'auto_awesome';
      } else if (popScore >= 60) {
        popBadgeClass = 'bg-secondary/20 text-secondary border-secondary/50';
        popIcon = 'trending_up';
      } else if (popScore >= 40) {
        popBadgeClass = 'bg-tertiary/20 text-tertiary border-tertiary/50';
        popIcon = 'queue_music';
      }

      item.innerHTML = `
        <div class="flex items-center gap-2.5 min-w-0">
          <div class="w-8 h-8 rounded-lg track-artwork-placeholder flex items-center justify-center shrink-0">
            <span class="material-symbols-outlined text-primary text-[18px]">music_note</span>
          </div>
          <div class="min-w-0">
            <div class="text-body-sm text-white font-semibold truncate">
              ${highlightMatch(track.name, query)}
            </div>
            <div class="text-[11px] text-on-surface-variant flex items-center gap-1.5 truncate">
              <span class="text-white/80">${highlightMatch(track.artist, query)}</span>
              <span>•</span>
              <span class="text-outline-variant truncate">${escapeHtml(track.album || '')}</span>
            </div>
          </div>
        </div>

        <div class="flex items-center gap-2 shrink-0">
          <span class="hidden sm:inline-block text-[10px] uppercase font-telemetry-label px-2 py-0.5 rounded bg-surface-container border border-outline-variant/30 text-on-surface-variant">
            ${escapeHtml(track.genre || 'pop')}
          </span>
          <div class="inline-flex items-center gap-1 px-2 py-0.5 rounded text-[11px] font-telemetry-label font-bold border ${popBadgeClass}">
            <span class="material-symbols-outlined text-[13px]">${popIcon}</span>
            <span>${popScore}</span>
          </div>
        </div>
      `;

      item.addEventListener('mouseenter', () => setSelectedIndex(idx));
      item.addEventListener('click', () => {
        loadTrackIntoStudio(track);
        fetchTrackPreviewAndPlay(track);
      });
      searchResultsList.appendChild(item);
    });
  }

  async function performSearch(query) {
    query = (query || '').trim();
    if (!query) {
      currentResults = TRENDING_SUGGESTIONS;
      renderResults(TRENDING_SUGGESTIONS, '');
      return;
    }

    let results = [];
    
    try {
      const resp = await fetch(`/api/search?q=${encodeURIComponent(query)}&limit=15`);
      if (resp.ok) {
        results = await resp.json();
      }
    } catch (e) {
      // Offline fallback
    }

    if (!results || results.length === 0) {
      if (localCatalog && localCatalog.length > 0) {
        const qLower = query.toLowerCase();
        results = localCatalog.filter(t => 
          (t.name && t.name.toLowerCase().includes(qLower)) || 
          (t.artist && t.artist.toLowerCase().includes(qLower)) ||
          (t.genre && t.genre.toLowerCase().includes(qLower))
        ).slice(0, 15);
      }
    }

    currentResults = results || [];
    renderResults(currentResults, query);
  }

  // Fetch streaming audio preview via /api/preview and start playback
  async function fetchTrackPreviewAndPlay(track) {
    if (!track) return;

    try {
      const resp = await fetch(`/api/preview?title=${encodeURIComponent(track.name)}&artist=${encodeURIComponent(track.artist)}&track_id=${encodeURIComponent(track.track_id || '')}`);
      if (resp.ok) {
        const data = await resp.json();
        if (data && data.preview_url) {
          audio.src = data.preview_url;
          audio.play().then(() => setPlayingState(true)).catch(e => console.log('Auto-play preview:', e));
          
          if (data.artwork_url && dockArtwork) {
            dockArtwork.innerHTML = `<img src="${data.artwork_url}" class="w-full h-full object-cover"/>`;
          }

          // Trigger automatic Librosa DSP signal analysis
          runLibrosaDspAnalysis(data.preview_url);
        } else {
          // If no external preview, analyze synthesized audio for track
          runLibrosaDspAnalysis(null);
        }
      }
    } catch (err) {
      console.log('Preview fetch notice:', err);
      runLibrosaDspAnalysis(null);
    }
  }

  // Ingest track into studio sliders, ML model & Spotify Player
  function loadTrackIntoStudio(track) {
    if (!track) return;

    currentLoadedTrack = track;
    activeTrackId = track.track_id || activeTrackId;

    // 1. Set continuous acoustic sliders
    if (sliders.danceability && track.danceability !== undefined) sliders.danceability.value = track.danceability;
    if (sliders.energy && track.energy !== undefined) sliders.energy.value = track.energy;
    if (sliders.loudness && track.loudness !== undefined) sliders.loudness.value = track.loudness;
    if (sliders.acousticness && track.acousticness !== undefined) sliders.acousticness.value = track.acousticness;
    if (sliders.instrumentalness && track.instrumentalness !== undefined) sliders.instrumentalness.value = track.instrumentalness;
    if (sliders.valence && track.valence !== undefined) sliders.valence.value = track.valence;
    if (sliders.tempo && track.tempo !== undefined) sliders.tempo.value = Math.round(track.tempo);
    if (sliders.speechiness && track.speechiness !== undefined) sliders.speechiness.value = track.speechiness;

    // 2. Set Genre dropdown
    if (selectGenre && track.genre) {
      let matched = false;
      for (let i = 0; i < selectGenre.options.length; i++) {
        if (selectGenre.options[i].value.toLowerCase() === track.genre.toLowerCase()) {
          selectGenre.selectedIndex = i;
          matched = true;
          break;
        }
      }
      if (!matched) {
        selectGenre.value = 'pop';
      }
      if (keyModeBadge && selectKey) {
        keyModeBadge.textContent = `${selectGenre.options[selectGenre.selectedIndex].text} • ${selectKey.options[selectKey.selectedIndex].text}`;
      }
    }

    // 3. Set Musical Key & Explicit
    if (selectKey && track.key !== undefined) {
      selectKey.value = String(track.key);
    }
    if (toggleExplicit && track.explicit !== undefined) {
      toggleExplicit.checked = Boolean(track.explicit);
    }

    // 4. Update Spotify Embed Player iframe
    if (spotifyPlayerIframe && activeTrackId) {
      spotifyPlayerIframe.src = `https://open.spotify.com/embed/track/${activeTrackId}?utm_source=generator&theme=0`;
    }
    const spotifyUrl = `https://open.spotify.com/track/${activeTrackId}`;
    if (bannerSpotifyLink) bannerSpotifyLink.href = spotifyUrl;
    if (spotifyDirectLink) spotifyDirectLink.href = spotifyUrl;
    if (dockSpotifyLink) dockSpotifyLink.href = spotifyUrl;

    // 5. Update Waveform Player Info
    if (playerTrackTitle) playerTrackTitle.textContent = track.name;
    if (playerArtistName) playerArtistName.textContent = track.artist;
    if (dockTrackTitle) dockTrackTitle.textContent = track.name;
    if (dockArtistName) dockArtistName.textContent = `${track.artist} • ${track.album || ''}`;
    if (spotifyDock) spotifyDock.classList.remove('collapsed');

    // 6. Reset preset pills active state
    document.querySelectorAll('.preset-pill').forEach(btn => btn.classList.remove('active'));

    // 7. Update UI & recompute prediction score
    updateUI();
    generateWaveform();

    // 8. Update and display Track Ingested Banner
    const predicted = lastPredictionData ? Math.round(lastPredictionData.predicted_popularity) : calculateScoreFallback();
    const actual = track.popularity !== undefined ? track.popularity : 50;
    const delta = predicted - actual;
    const accuracy = Math.max(0, 100 - Math.abs(delta));

    if (loadedTrackBanner) {
      loadedTrackBanner.classList.remove('hidden');
      if (loadedTrackTitle) loadedTrackTitle.textContent = track.name;
      if (loadedTrackArtist) loadedTrackArtist.textContent = track.artist;
      if (loadedTrackAlbum) loadedTrackAlbum.textContent = track.album || 'Single';
      if (loadedTrackGenre) loadedTrackGenre.textContent = track.genre || 'pop';
      if (loadedActualScore) loadedActualScore.textContent = actual;
      if (loadedPredictedScore) loadedPredictedScore.textContent = predicted;
      if (loadedAccuracyBadge) {
        const deltaSign = delta > 0 ? `+${delta}` : `${delta}`;
        loadedAccuracyBadge.textContent = `Δ ${deltaSign} pts (${accuracy.toFixed(1)}% match)`;
        if (Math.abs(delta) <= 5) {
          loadedAccuracyBadge.className = 'bg-primary/20 text-primary border border-primary/40 text-[10px] font-telemetry-label px-1.5 py-0.5 rounded font-bold';
        } else if (Math.abs(delta) <= 12) {
          loadedAccuracyBadge.className = 'bg-secondary/20 text-secondary border border-secondary/40 text-[10px] font-telemetry-label px-1.5 py-0.5 rounded';
        } else {
          loadedAccuracyBadge.className = 'bg-tertiary/20 text-tertiary border border-tertiary/40 text-[10px] font-telemetry-label px-1.5 py-0.5 rounded';
        }
      }
    }

    // 9. Update search input text and close dropdown
    if (searchInput) searchInput.value = `${track.name} — ${track.artist}`;
    if (clearSearchBtn) clearSearchBtn.classList.remove('hidden');
    if (searchResultsDropdown) searchResultsDropdown.classList.add('hidden');
  }

  // Search Input Events
  if (searchInput) {
    searchInput.addEventListener('input', (e) => {
      const val = e.target.value;
      if (clearSearchBtn) {
        if (val.trim()) clearSearchBtn.classList.remove('hidden');
        else clearSearchBtn.classList.add('hidden');
      }

      clearTimeout(debounceTimer);
      debounceTimer = setTimeout(() => {
        performSearch(val);
      }, 120);
    });

    searchInput.addEventListener('focus', () => {
      performSearch(searchInput.value);
    });

    searchInput.addEventListener('keydown', (e) => {
      if (!searchResultsDropdown || searchResultsDropdown.classList.contains('hidden')) {
        if (e.key === 'ArrowDown') {
          performSearch(searchInput.value);
          e.preventDefault();
        }
        return;
      }

      const items = searchResultsList ? searchResultsList.querySelectorAll('.search-result-item') : [];
      if (items.length === 0) return;

      if (e.key === 'ArrowDown') {
        e.preventDefault();
        const next = selectedIndex < items.length - 1 ? selectedIndex + 1 : 0;
        setSelectedIndex(next);
      } else if (e.key === 'ArrowUp') {
        e.preventDefault();
        const prev = selectedIndex > 0 ? selectedIndex - 1 : items.length - 1;
        setSelectedIndex(prev);
      } else if (e.key === 'Enter') {
        e.preventDefault();
        if (selectedIndex >= 0 && selectedIndex < currentResults.length) {
          const t = currentResults[selectedIndex];
          loadTrackIntoStudio(t);
          fetchTrackPreviewAndPlay(t);
        }
      } else if (e.key === 'Escape') {
        searchResultsDropdown.classList.add('hidden');
      }
    });
  }

  // Clear search button
  if (clearSearchBtn) {
    clearSearchBtn.addEventListener('click', () => {
      if (searchInput) {
        searchInput.value = '';
        searchInput.focus();
      }
      clearSearchBtn.classList.add('hidden');
      if (searchResultsDropdown) searchResultsDropdown.classList.add('hidden');
    });
  }

  // Close dropdown on outside click
  document.addEventListener('click', (e) => {
    if (searchResultsDropdown && !searchResultsDropdown.contains(e.target) && e.target !== searchInput) {
      searchResultsDropdown.classList.add('hidden');
    }
  });

  // Reset Loaded Track Button
  if (clearLoadedTrackBtn) {
    clearLoadedTrackBtn.addEventListener('click', () => {
      currentLoadedTrack = null;
      if (loadedTrackBanner) loadedTrackBanner.classList.add('hidden');
      if (searchInput) searchInput.value = '';
      if (clearSearchBtn) clearSearchBtn.classList.add('hidden');
      audio.pause();
      setPlayingState(false);
      const popPreset = document.querySelector('[data-preset="pop"]');
      if (popPreset) popPreset.click();
    });
  }

  // ============================================================================
  // 8. CONTROLS & EVENT LISTENERS
  // ============================================================================
  Object.values(sliders).forEach(slider => {
    if (slider) {
      slider.addEventListener('input', () => {
        updateUI();
        document.querySelectorAll('.preset-pill').forEach(btn => btn.classList.remove('active'));
      });
    }
  });

  if (toggleExplicit) toggleExplicit.addEventListener('change', updateUI);
  if (selectGenre) {
    selectGenre.addEventListener('change', () => {
      if (keyModeBadge && selectKey) {
        keyModeBadge.textContent = `${selectGenre.options[selectGenre.selectedIndex].text} • ${selectKey.options[selectKey.selectedIndex].text}`;
      }
      updateUI();
    });
  }
  if (selectKey) selectKey.addEventListener('change', updateUI);

  // Preset Buttons
  document.querySelectorAll('.preset-pill').forEach((pill, idx) => {
    pill.addEventListener('click', () => {
      document.querySelectorAll('.preset-pill').forEach(btn => btn.classList.remove('active'));
      pill.classList.add('active');
      currentPresetIndex = idx;

      const presetKey = pill.getAttribute('data-preset');
      const p = PRESETS[presetKey];
      if (p) {
        currentLoadedTrack = null;
        if (loadedTrackBanner) loadedTrackBanner.classList.add('hidden');

        sliders.danceability.value = p.danceability;
        sliders.energy.value = p.energy;
        sliders.loudness.value = p.loudness;
        sliders.acousticness.value = p.acousticness;
        sliders.instrumentalness.value = p.instrumentalness;
        sliders.valence.value = p.valence;
        sliders.tempo.value = p.tempo;
        sliders.speechiness.value = p.speechiness;
        if (selectGenre) selectGenre.value = p.genre;
        if (selectKey) selectKey.value = p.key;
        if (toggleExplicit) toggleExplicit.checked = p.explicit;
        if (keyModeBadge && selectGenre && selectKey) {
          keyModeBadge.textContent = `${selectGenre.options[selectGenre.selectedIndex].text} • ${selectKey.options[selectKey.selectedIndex].text}`;
        }

        // Update Spotify Player Embed for Preset
        if (p.track_id && spotifyPlayerIframe) {
          activeTrackId = p.track_id;
          spotifyPlayerIframe.src = `https://open.spotify.com/embed/track/${p.track_id}?utm_source=generator&theme=0`;
          const spotifyUrl = `https://open.spotify.com/track/${p.track_id}`;
          if (bannerSpotifyLink) bannerSpotifyLink.href = spotifyUrl;
          if (spotifyDirectLink) spotifyDirectLink.href = spotifyUrl;
          if (dockSpotifyLink) dockSpotifyLink.href = spotifyUrl;
        }

        if (playerTrackTitle) playerTrackTitle.textContent = p.name;
        if (playerArtistName) playerArtistName.textContent = p.artist;
        if (dockTrackTitle) dockTrackTitle.textContent = p.name;
        if (dockArtistName) dockArtistName.textContent = p.artist;

        generateWaveform();
        updateUI();
      }
    });
  });

  // Dock Previous & Next Preset navigation
  if (dockPrevBtn) {
    dockPrevBtn.addEventListener('click', () => {
      currentPresetIndex = (currentPresetIndex - 1 + PRESET_KEYS.length) % PRESET_KEYS.length;
      const key = PRESET_KEYS[currentPresetIndex];
      const btn = document.querySelector(`[data-preset="${key}"]`);
      if (btn) btn.click();
    });
  }

  if (dockNextBtn) {
    dockNextBtn.addEventListener('click', () => {
      currentPresetIndex = (currentPresetIndex + 1) % PRESET_KEYS.length;
      const key = PRESET_KEYS[currentPresetIndex];
      const btn = document.querySelector(`[data-preset="${key}"]`);
      if (btn) btn.click();
    });
  }

  // Reset Defaults Button
  const resetBtn = document.getElementById('resetDefaultsBtn');
  if (resetBtn) {
    resetBtn.addEventListener('click', () => {
      const popPreset = document.querySelector('[data-preset="pop"]');
      if (popPreset) popPreset.click();
    });
  }

  // Compute CTA Button
  const runPredictBtn = document.getElementById('runPredictBtn');
  if (runPredictBtn) {
    runPredictBtn.addEventListener('click', () => {
      updateUI();
      generateWaveform();
      if (scoreNumber) {
        scoreNumber.classList.add('scale-110');
        setTimeout(() => scoreNumber.classList.remove('scale-110'), 200);
      }
    });
  }

  // Export Telemetry
  const exportBtn = document.getElementById('exportBtn');
  if (exportBtn) {
    exportBtn.addEventListener('click', () => {
      const data = {
        score: scoreNumber ? scoreNumber.textContent : 0,
        danceability: sliders.danceability ? sliders.danceability.value : 0,
        energy: sliders.energy ? sliders.energy.value : 0,
        loudness: sliders.loudness ? sliders.loudness.value : 0,
        acousticness: sliders.acousticness ? sliders.acousticness.value : 0,
        valence: sliders.valence ? sliders.valence.value : 0,
        tempo: sliders.tempo ? sliders.tempo.value : 0,
        genre: selectGenre ? selectGenre.value : 'pop',
        loadedTrack: currentLoadedTrack ? {
          track_id: currentLoadedTrack.track_id,
          name: currentLoadedTrack.name,
          artist: currentLoadedTrack.artist,
          actualPopularity: currentLoadedTrack.popularity
        } : null,
        timestamp: new Date().toISOString()
      };
      const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `soundlab_telemetry_${Date.now()}.json`;
      a.click();
      URL.revokeObjectURL(url);
    });
  }

  // Keyboard shortcut: ⌘K or Ctrl+K to focus search input
  window.addEventListener('keydown', (e) => {
    if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
      e.preventDefault();
      if (searchInput) {
        searchInput.focus();
        searchInput.select();
      }
    }
  });

  // ============================================================================
  // 8. AUTOMATIC AUDIO SIGNAL ANALYSIS ENGINE (LIBROSA DSP)
  // ============================================================================
  const uploadAudioBtn = document.getElementById('uploadAudioBtn');
  const audioFileInput = document.getElementById('audioFileInput');
  const runDspAnalysisBtn = document.getElementById('runDspAnalysisBtn');

  async function runLibrosaDspAnalysis(previewUrl = null, fileBytes = null) {
    const signalBadge = document.getElementById('dspSignalSourceBadge');
    if (signalBadge) signalBadge.textContent = "Analyzing Librosa DSP...";

    try {
      let resp;
      if (fileBytes) {
        resp = await fetch('/api/analyze_audio', {
          method: 'POST',
          headers: { 'Content-Type': 'application/octet-stream' },
          body: fileBytes
        });
      } else {
        resp = await fetch('/api/analyze_audio', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ audio_url: previewUrl || (audio ? audio.src : null) })
        });
      }

      if (resp.ok) {
        const dsp = await resp.json();
        updateDspUI(dsp);
      } else {
        if (signalBadge) signalBadge.textContent = "Source: Signal Simulated (DSP Ready)";
      }
    } catch (err) {
      console.log('DSP Analysis notice:', err);
      if (signalBadge) signalBadge.textContent = "Source: Signal Ready";
    }
  }

  function updateDspUI(dsp) {
    if (!dsp) return;
    const tempoEl = document.getElementById('dspTempoVal');
    const beatsEl = document.getElementById('dspBeatCount');
    const rmsMeanEl = document.getElementById('dspRmsMeanVal');
    const rmsPeakEl = document.getElementById('dspRmsPeak');
    const loudEl = document.getElementById('dspLoudnessVal');
    const centEl = document.getElementById('dspCentroidVal');
    const bwEl = document.getElementById('dspBandwidthVal');
    const rollEl = document.getElementById('dspRolloffVal');
    const zcrEl = document.getElementById('dspZcrVal');
    const regEl = document.getElementById('dspRegularityVal');
    const signalBadge = document.getElementById('dspSignalSourceBadge');

    if (tempoEl && dsp.tempo_bpm !== undefined) tempoEl.textContent = dsp.tempo_bpm.toFixed(1);
    if (beatsEl && dsp.beat_count !== undefined) beatsEl.textContent = `${dsp.beat_count} detected beat frames`;
    if (rmsMeanEl && dsp.rms_energy_mean !== undefined) rmsMeanEl.textContent = dsp.rms_energy_mean.toFixed(3);
    if (rmsPeakEl && dsp.rms_energy_peak !== undefined) rmsPeakEl.textContent = `Peak: ${dsp.rms_energy_peak.toFixed(3)} amplitude`;
    if (loudEl && dsp.estimated_loudness_db !== undefined) loudEl.textContent = dsp.estimated_loudness_db.toFixed(1);
    if (centEl && dsp.spectral_centroid_hz !== undefined) centEl.textContent = Math.round(dsp.spectral_centroid_hz).toLocaleString();
    if (bwEl && dsp.spectral_bandwidth_hz !== undefined) bwEl.textContent = Math.round(dsp.spectral_bandwidth_hz).toLocaleString();
    if (rollEl && dsp.spectral_rolloff_hz !== undefined) rollEl.textContent = Math.round(dsp.spectral_rolloff_hz).toLocaleString();
    if (zcrEl && dsp.zero_crossing_rate !== undefined) zcrEl.textContent = dsp.zero_crossing_rate.toFixed(4);
    if (regEl && dsp.rhythm_regularity_proxy !== undefined) regEl.textContent = dsp.rhythm_regularity_proxy.toFixed(2);

    if (signalBadge) {
      signalBadge.textContent = dsp.signal_source === 'soundfile_decoder' 
        ? `Source: Decoded Audio (${dsp.sample_rate}Hz • ${dsp.duration_seconds}s)`
        : `Source: Librosa DSP Stream (${dsp.sample_rate}Hz)`;
    }
  }

  // Audio Upload & Trigger listeners
  if (uploadAudioBtn && audioFileInput) {
    uploadAudioBtn.addEventListener('click', () => audioFileInput.click());
    audioFileInput.addEventListener('change', (e) => {
      const file = e.target.files[0];
      if (!file) return;
      const reader = new FileReader();
      reader.onload = (event) => {
        runLibrosaDspAnalysis(null, event.target.result);
      };
      reader.readAsArrayBuffer(file);
    });
  }

  if (runDspAnalysisBtn) {
    runDspAnalysisBtn.addEventListener('click', () => {
      runLibrosaDspAnalysis(audio ? audio.src : null);
    });
  }

  // Section Tabs Active Toggle & ScrollSpy
  const navTabs = document.querySelectorAll('header nav a[href^="#"]');
  navTabs.forEach(link => {
    link.addEventListener('click', () => {
      navTabs.forEach(l => l.classList.remove('active'));
      link.classList.add('active');
    });
  });

  // IntersectionObserver ScrollSpy to update active header nav as user scrolls
  const observedSectionIds = ['predictor', 'audio-dsp', 'retention-intelligence', 'correlation'];
  const scrollObserver = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        const id = entry.target.id;
        navTabs.forEach(tab => {
          if (tab.getAttribute('href') === `#${id}`) {
            navTabs.forEach(t => t.classList.remove('active'));
            tab.classList.add('active');
          }
        });
      }
    });
  }, { rootMargin: '-20% 0px -70% 0px' });

  observedSectionIds.forEach(id => {
    const el = document.getElementById(id);
    if (el) scrollObserver.observe(el);
  });

  // Initial update & trigger default DSP analysis
  updateUI();
  runLibrosaDspAnalysis(null);
});

