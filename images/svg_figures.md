# SVG Figures for Presentation
## Scalable Vector Graphics for Enhanced Visual Communication

---

## SVG Figure 1: Interactive Model Performance Comparison

```svg
<svg width="800" height="600" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <linearGradient id="blueGradient" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" style="stop-color:#337ab7;stop-opacity:1" />
      <stop offset="100%" style="stop-color:#5bc0de;stop-opacity:1" />
    </linearGradient>
    <filter id="shadow" x="-20%" y="-20%" width="140%" height="140%">
      <feDropShadow dx="3" dy="3" stdDeviation="3" flood-color="#000" flood-opacity="0.3"/>
    </filter>
  </defs>
  
  <!-- Background -->
  <rect width="800" height="600" fill="#f8f9fa"/>
  
  <!-- Title -->
  <text x="400" y="40" text-anchor="middle" font-family="Arial, sans-serif" font-size="24" font-weight="bold" fill="#333">
    Model Performance Comparison
  </text>
  
  <!-- Performance bars -->
  <g id="performance-bars">
    <!-- TruncatedModel -->
    <rect x="50" y="100" width="120" height="30" fill="url(#blueGradient)" filter="url(#shadow)"/>
    <text x="60" y="120" font-family="Arial" font-size="12" fill="white">TruncatedModel</text>
    <text x="180" y="120" font-family="Arial" font-size="14" fill="#333">85.3%</text>
    
    <!-- SegmentationModel -->
    <rect x="50" y="150" width="140" height="30" fill="url(#blueGradient)" filter="url(#shadow)"/>
    <text x="60" y="170" font-family="Arial" font-size="12" fill="white">SegmentationModel</text>
    <text x="200" y="170" font-family="Arial" font-size="14" fill="#333">89.7%</text>
    
    <!-- ClassificationModel -->
    <rect x="50" y="200" width="160" height="30" fill="url(#blueGradient)" filter="url(#shadow)"/>
    <text x="60" y="220" font-family="Arial" font-size="12" fill="white">ClassificationModel</text>
    <text x="220" y="220" font-family="Arial" font-size="14" fill="#333">92.1%</text>
    
    <!-- Multi-Modal Multi-Scale -->
    <rect x="50" y="250" width="180" height="30" fill="url(#blueGradient)" filter="url(#shadow)"/>
    <text x="60" y="270" font-family="Arial" font-size="12" fill="white">Multi-Modal Multi-Scale</text>
    <text x="240" y="270" font-family="Arial" font-size="14" fill="#333">94.8%</text>
    
    <!-- Multi-Modal Multi-Fusion -->
    <rect x="50" y="300" width="170" height="30" fill="url(#blueGradient)" filter="url(#shadow)"/>
    <text x="60" y="320" font-family="Arial" font-size="12" fill="white">Multi-Modal Multi-Fusion</text>
    <text x="230" y="320" font-family="Arial" font-size="14" fill="#333">93.5%</text>
  </g>
  
  <!-- Legend -->
  <rect x="400" y="100" width="350" height="200" fill="white" stroke="#ddd" stroke-width="1" rx="5"/>
  <text x="420" y="125" font-family="Arial" font-size="16" font-weight="bold" fill="#333">Performance Metrics</text>
  
  <!-- Metric details -->
  <text x="420" y="150" font-family="Arial" font-size="12" fill="#666">• Accuracy: Classification correctness</text>
  <text x="420" y="170" font-family="Arial" font-size="12" fill="#666">• Training Time: Convergence speed</text>
  <text x="420" y="190" font-family="Arial" font-size="12" fill="#666">• Memory Usage: Resource efficiency</text>
  <text x="420" y="210" font-family="Arial" font-size="12" fill="#666">• Interpretability: XAI compatibility</text>
  <text x="420" y="230" font-family="Arial" font-size="12" fill="#666">• Scalability: Production readiness</text>
  
  <!-- Best performer highlight -->
  <circle cx="35" cy="265" r="8" fill="#d9534f"/>
  <text x="420" y="270" font-family="Arial" font-size="12" font-weight="bold" fill="#d9534f">★ Best Overall Performance</text>
</svg>
```

---

## SVG Figure 2: XAI Attribution Visualization Concept

```svg
<svg width="900" height="500" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <radialGradient id="heatmap" cx="50%" cy="50%" r="50%">
      <stop offset="0%" style="stop-color:#ff0000;stop-opacity:0.8" />
      <stop offset="30%" style="stop-color:#ff8800;stop-opacity:0.6" />
      <stop offset="60%" style="stop-color:#ffff00;stop-opacity:0.4" />
      <stop offset="100%" style="stop-color:#ffffff;stop-opacity:0.1" />
    </radialGradient>
    <pattern id="grid" width="20" height="20" patternUnits="userSpaceOnUse">
      <path d="M 20 0 L 0 0 0 20" fill="none" stroke="#e0e0e0" stroke-width="1"/>
    </pattern>
  </defs>
  
  <!-- Background -->
  <rect width="900" height="500" fill="#f8f9fa"/>
  
  <!-- Title -->
  <text x="450" y="30" text-anchor="middle" font-family="Arial, sans-serif" font-size="20" font-weight="bold" fill="#333">
    XAI Attribution Visualization Process
  </text>
  
  <!-- Original Image Section -->
  <g id="original-image">
    <rect x="50" y="70" width="200" height="150" fill="url(#grid)" stroke="#666" stroke-width="2"/>
    <text x="150" y="95" text-anchor="middle" font-family="Arial" font-size="14" fill="#333">Original Image</text>
    
    <!-- Simulated image content -->
    <rect x="70" y="110" width="60" height="40" fill="#4CAF50" opacity="0.7"/>
    <rect x="140" y="120" width="40" height="30" fill="#2196F3" opacity="0.7"/>
    <rect x="90" y="160" width="80" height="20" fill="#FF9800" opacity="0.7"/>
    <text x="150" y="240" text-anchor="middle" font-family="Arial" font-size="12" fill="#666">Input Data</text>
  </g>
  
  <!-- Arrow 1 -->
  <path d="M 270 145 L 320 145" stroke="#333" stroke-width="3" marker-end="url(#arrowhead)"/>
  <text x="295" y="135" text-anchor="middle" font-family="Arial" font-size="10" fill="#666">Captum</text>
  
  <!-- Attribution Heatmap Section -->
  <g id="attribution-heatmap">
    <rect x="340" y="70" width="200" height="150" fill="white" stroke="#666" stroke-width="2"/>
    <text x="440" y="95" text-anchor="middle" font-family="Arial" font-size="14" fill="#333">Attribution Heatmap</text>
    
    <!-- Heatmap visualization -->
    <circle cx="400" cy="130" r="30" fill="url(#heatmap)"/>
    <circle cx="480" cy="150" r="20" fill="url(#heatmap)" opacity="0.7"/>
    <circle cx="420" cy="180" r="25" fill="url(#heatmap)" opacity="0.5"/>
    <text x="440" y="240" text-anchor="middle" font-family="Arial" font-size="12" fill="#666">Feature Attribution</text>
  </g>
  
  <!-- Arrow 2 -->
  <path d="M 560 145 L 610 145" stroke="#333" stroke-width="3" marker-end="url(#arrowhead)"/>
  <text x="585" y="135" text-anchor="middle" font-family="Arial" font-size="10" fill="#666">Overlay</text>
  
  <!-- Overlay Visualization Section -->
  <g id="overlay-viz">
    <rect x="630" y="70" width="200" height="150" fill="white" stroke="#666" stroke-width="2"/>
    <text x="730" y="95" text-anchor="middle" font-family="Arial" font-size="14" fill="#333">Interpretable Overlay</text>
    
    <!-- Combined visualization -->
    <rect x="650" y="110" width="60" height="40" fill="#4CAF50" opacity="0.7"/>
    <circle cx="680" cy="130" r="15" fill="#ff0000" opacity="0.6"/>
    
    <rect x="720" y="120" width="40" height="30" fill="#2196F3" opacity="0.7"/>
    <circle cx="740" cy="135" r="10" fill="#ff8800" opacity="0.6"/>
    
    <rect x="670" y="160" width="80" height="20" fill="#FF9800" opacity="0.7"/>
    <circle cx="710" cy="170" r="12" fill="#ffff00" opacity="0.6"/>
    
    <text x="730" y="240" text-anchor="middle" font-family="Arial" font-size="12" fill="#666">Explained Predictions</text>
  </g>
  
  <!-- Process Steps -->
  <g id="process-steps">
    <rect x="50" y="300" width="780" height="150" fill="white" stroke="#ddd" stroke-width="1" rx="10"/>
    <text x="440" y="325" text-anchor="middle" font-family="Arial" font-size="16" font-weight="bold" fill="#333">
      Integrated Gradients Analysis Steps
    </text>
    
    <!-- Step boxes -->
    <rect x="80" y="340" width="140" height="80" fill="#e3f2fd" stroke="#2196F3" stroke-width="1" rx="5"/>
    <text x="150" y="365" text-anchor="middle" font-family="Arial" font-size="12" font-weight="bold" fill="#1976D2">
      1. Input Processing
    </text>
    <text x="150" y="380" text-anchor="middle" font-family="Arial" font-size="10" fill="#666">
      Prepare image data
    </text>
    <text x="150" y="395" text-anchor="middle" font-family="Arial" font-size="10" fill="#666">
      Set baseline reference
    </text>
    <text x="150" y="410" text-anchor="middle" font-family="Arial" font-size="10" fill="#666">
      Define integration path
    </text>
    
    <rect x="250" y="340" width="140" height="80" fill="#f3e5f5" stroke="#9C27B0" stroke-width="1" rx="5"/>
    <text x="320" y="365" text-anchor="middle" font-family="Arial" font-size="12" font-weight="bold" fill="#7B1FA2">
      2. Gradient Computation
    </text>
    <text x="320" y="380" text-anchor="middle" font-family="Arial" font-size="10" fill="#666">
      Calculate gradients
    </text>
    <text x="320" y="395" text-anchor="middle" font-family="Arial" font-size="10" fill="#666">
      Integrate attribution
    </text>
    <text x="320" y="410" text-anchor="middle" font-family="Arial" font-size="10" fill="#666">
      Generate importance
    </text>
    
    <rect x="420" y="340" width="140" height="80" fill="#e8f5e8" stroke="#4CAF50" stroke-width="1" rx="5"/>
    <text x="490" y="365" text-anchor="middle" font-family="Arial" font-size="12" font-weight="bold" fill="#388E3C">
      3. Visualization
    </text>
    <text x="490" y="380" text-anchor="middle" font-family="Arial" font-size="10" fill="#666">
      Create heatmaps
    </text>
    <text x="490" y="395" text-anchor="middle" font-family="Arial" font-size="10" fill="#666">
      Apply color mapping
    </text>
    <text x="490" y="410" text-anchor="middle" font-family="Arial" font-size="10" fill="#666">
      Generate overlays
    </text>
    
    <rect x="590" y="340" width="140" height="80" fill="#fff3e0" stroke="#FF9800" stroke-width="1" rx="5"/>
    <text x="660" y="365" text-anchor="middle" font-family="Arial" font-size="12" font-weight="bold" fill="#F57C00">
      4. Analysis
    </text>
    <text x="660" y="380" text-anchor="middle" font-family="Arial" font-size="10" fill="#666">
      Interpret patterns
    </text>
    <text x="660" y="395" text-anchor="middle" font-family="Arial" font-size="10" fill="#666">
      Validate results
    </text>
    <text x="660" y="410" text-anchor="middle" font-family="Arial" font-size="10" fill="#666">
      Extract insights
    </text>
  </g>
  
  <!-- Arrow marker definition -->
  <defs>
    <marker id="arrowhead" markerWidth="10" markerHeight="7" refX="10" refY="3.5" orient="auto">
      <polygon points="0 0, 10 3.5, 0 7" fill="#333"/>
    </marker>
  </defs>
</svg>
```

---

## SVG Figure 3: Binary Classifier Strategy Visualization

```svg
<svg width="1000" height="600" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <linearGradient id="redGradient" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" style="stop-color:#d9534f;stop-opacity:1" />
      <stop offset="100%" style="stop-color:#f0ad4e;stop-opacity:1" />
    </linearGradient>
    <linearGradient id="greenGradient" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" style="stop-color:#5cb85c;stop-opacity:1" />
      <stop offset="100%" style="stop-color:#5bc0de;stop-opacity:1" />
    </linearGradient>
  </defs>
  
  <!-- Background -->
  <rect width="1000" height="600" fill="#f8f9fa"/>
  
  <!-- Title -->
  <text x="500" y="40" text-anchor="middle" font-family="Arial, sans-serif" font-size="24" font-weight="bold" fill="#333">
    Binary Classifier Strategy: Current vs Proposed Approach
  </text>
  
  <!-- Current Approach Section -->
  <g id="current-approach">
    <rect x="50" y="80" width="400" height="450" fill="white" stroke="#d9534f" stroke-width="3" rx="10"/>
    <text x="250" y="110" text-anchor="middle" font-family="Arial" font-size="18" font-weight="bold" fill="#d9534f">
      Current Multi-Class Approach
    </text>
    
    <!-- Data Input -->
    <rect x="80" y="140" width="120" height="60" fill="#e6f3ff" stroke="#337ab7" stroke-width="2" rx="5"/>
    <text x="140" y="165" text-anchor="middle" font-family="Arial" font-size="12" fill="#333">Multi-Class</text>
    <text x="140" y="180" text-anchor="middle" font-family="Arial" font-size="12" fill="#333">Dataset</text>
    <text x="140" y="195" text-anchor="middle" font-family="Arial" font-size="10" fill="#666">(5 classes)</text>
    
    <!-- Single Model -->
    <rect x="150" y="240" width="140" height="80" fill="url(#redGradient)" rx="10"/>
    <text x="220" y="270" text-anchor="middle" font-family="Arial" font-size="14" font-weight="bold" fill="white">
      Single Complex
    </text>
    <text x="220" y="290" text-anchor="middle" font-family="Arial" font-size="14" font-weight="bold" fill="white">
      Multi-Class
    </text>
    <text x="220" y="310" text-anchor="middle" font-family="Arial" font-size="14" font-weight="bold" fill="white">
      Classifier
    </text>
    
    <!-- Output -->
    <rect x="160" y="360" width="120" height="60" fill="#ffe6e6" stroke="#d9534f" stroke-width="2" rx="5"/>
    <text x="220" y="385" text-anchor="middle" font-family="Arial" font-size="12" fill="#333">Classes:</text>
    <text x="220" y="400" text-anchor="middle" font-family="Arial" font-size="10" fill="#666">1, 2, 3, 4, 5</text>
    
    <!-- Challenges -->
    <rect x="70" y="450" width="320" height="70" fill="#f2dede" stroke="#d9534f" stroke-width="1" rx="5"/>
    <text x="230" y="470" text-anchor="middle" font-family="Arial" font-size="12" font-weight="bold" fill="#a94442">
      Challenges:
    </text>
    <text x="90" y="490" font-family="Arial" font-size="10" fill="#a94442">• Complex decision boundaries</text>
    <text x="90" y="505" font-family="Arial" font-size="10" fill="#a94442">• Class imbalance sensitivity</text>
    <text x="270" y="490" font-family="Arial" font-size="10" fill="#a94442">• Difficult optimization</text>
    <text x="270" y="505" font-family="Arial" font-size="10" fill="#a94442">• Lower interpretability</text>
    
    <!-- Arrows -->
    <path d="M 140 210 L 190 230" stroke="#333" stroke-width="2" marker-end="url(#arrowhead)"/>
    <path d="M 220 330 L 220 350" stroke="#333" stroke-width="2" marker-end="url(#arrowhead)"/>
  </g>
  
  <!-- Proposed Approach Section -->
  <g id="proposed-approach">
    <rect x="550" y="80" width="400" height="450" fill="white" stroke="#5cb85c" stroke-width="3" rx="10"/>
    <text x="750" y="110" text-anchor="middle" font-family="Arial" font-size="18" font-weight="bold" fill="#5cb85c">
      Proposed Binary Classifier Approach
    </text>
    
    <!-- Data Input -->
    <rect x="580" y="140" width="120" height="60" fill="#e6f3ff" stroke="#337ab7" stroke-width="2" rx="5"/>
    <text x="640" y="165" text-anchor="middle" font-family="Arial" font-size="12" fill="#333">Binary</text>
    <text x="640" y="180" text-anchor="middle" font-family="Arial" font-size="12" fill="#333">Datasets</text>
    <text x="640" y="195" text-anchor="middle" font-family="Arial" font-size="10" fill="#666">(Class vs No-Class)</text>
    
    <!-- Multiple Binary Classifiers -->
    <rect x="580" y="240" width="80" height="60" fill="url(#greenGradient)" rx="5"/>
    <text x="620" y="265" text-anchor="middle" font-family="Arial" font-size="10" font-weight="bold" fill="white">Binary</text>
    <text x="620" y="280" text-anchor="middle" font-family="Arial" font-size="10" font-weight="bold" fill="white">Classifier 1</text>
    
    <rect x="680" y="240" width="80" height="60" fill="url(#greenGradient)" rx="5"/>
    <text x="720" y="265" text-anchor="middle" font-family="Arial" font-size="10" font-weight="bold" fill="white">Binary</text>
    <text x="720" y="280" text-anchor="middle" font-family="Arial" font-size="10" font-weight="bold" fill="white">Classifier 2</text>
    
    <rect x="780" y="240" width="80" height="60" fill="url(#greenGradient)" rx="5"/>
    <text x="820" y="265" text-anchor="middle" font-family="Arial" font-size="10" font-weight="bold" fill="white">Binary</text>
    <text x="820" y="280" text-anchor="middle" font-family="Arial" font-size="10" font-weight="bold" fill="white">Classifier 3</text>
    
    <text x="870" y="270" text-anchor="middle" font-family="Arial" font-size="20" fill="#666">...</text>
    
    <!-- Ensemble Decision -->
    <rect x="680" y="320" width="100" height="40" fill="#d4edda" stroke="#5cb85c" stroke-width="2" rx="5"/>
    <text x="730" y="340" text-anchor="middle" font-family="Arial" font-size="12" fill="#333">Ensemble</text>
    <text x="730" y="355" text-anchor="middle" font-family="Arial" font-size="12" fill="#333">Decision</text>
    
    <!-- Output -->
    <rect x="680" y="380" width="100" height="40" fill="#e6ffe6" stroke="#5cb85c" stroke-width="2" rx="5"/>
    <text x="730" y="400" text-anchor="middle" font-family="Arial" font-size="12" fill="#333">Anomaly Type</text>
    <text x="730" y="415" text-anchor="middle" font-family="Arial" font-size="12" fill="#333">Detection</text>
    
    <!-- Benefits -->
    <rect x="570" y="450" width="320" height="70" fill="#dff0d8" stroke="#5cb85c" stroke-width="1" rx="5"/>
    <text x="730" y="470" text-anchor="middle" font-family="Arial" font-size="12" font-weight="bold" fill="#3c763d">
      Benefits:
    </text>
    <text x="590" y="490" font-family="Arial" font-size="10" fill="#3c763d">• Simpler decision boundaries</text>
    <text x="590" y="505" font-family="Arial" font-size="10" fill="#3c763d">• Better class balance handling</text>
    <text x="770" y="490" font-family="Arial" font-size="10" fill="#3c763d">• Parallel optimization</text>
    <text x="770" y="505" font-family="Arial" font-size="10" fill="#3c763d">• Enhanced interpretability</text>
    
    <!-- Arrows -->
    <path d="M 640 210 L 620 230" stroke="#333" stroke-width="2" marker-end="url(#arrowhead)"/>
    <path d="M 640 210 L 720 230" stroke="#333" stroke-width="2" marker-end="url(#arrowhead)"/>
    <path d="M 640 210 L 820 230" stroke="#333" stroke-width="2" marker-end="url(#arrowhead)"/>
    
    <path d="M 620 310 L 700 320" stroke="#333" stroke-width="2" marker-end="url(#arrowhead)"/>
    <path d="M 720 310 L 720 320" stroke="#333" stroke-width="2" marker-end="url(#arrowhead)"/>
    <path d="M 820 310 L 750 320" stroke="#333" stroke-width="2" marker-end="url(#arrowhead)"/>
    
    <path d="M 730 370 L 730 375" stroke="#333" stroke-width="2" marker-end="url(#arrowhead)"/>
  </g>
  
  <!-- Comparison Arrow -->
  <path d="M 470 300 L 530 300" stroke="#333" stroke-width="5" marker-end="url(#bigArrowhead)"/>
  <text x="500" y="290" text-anchor="middle" font-family="Arial" font-size="12" font-weight="bold" fill="#333">
    Proposed
  </text>
  <text x="500" y="320" text-anchor="middle" font-family="Arial" font-size="12" font-weight="bold" fill="#333">
    Improvement
  </text>
  
  <!-- Arrow marker definitions -->
  <defs>
    <marker id="arrowhead" markerWidth="10" markerHeight="7" refX="10" refY="3.5" orient="auto">
      <polygon points="0 0, 10 3.5, 0 7" fill="#333"/>
    </marker>
    <marker id="bigArrowhead" markerWidth="15" markerHeight="12" refX="15" refY="6" orient="auto">
      <polygon points="0 0, 15 6, 0 12" fill="#333"/>
    </marker>
  </defs>
</svg>
```

---

## SVG Figure 4: Production Deployment Flow

```svg
<svg width="1200" height="700" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <linearGradient id="blueFlow" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" style="stop-color:#337ab7;stop-opacity:1" />
      <stop offset="100%" style="stop-color:#5bc0de;stop-opacity:1" />
    </linearGradient>
    <filter id="glow" x="-50%" y="-50%" width="200%" height="200%">
      <feGaussianBlur stdDeviation="3" result="coloredBlur"/>
      <feMerge> 
        <feMergeNode in="coloredBlur"/>
        <feMergeNode in="SourceGraphic"/>
      </feMerge>
    </filter>
  </defs>
  
  <!-- Background -->
  <rect width="1200" height="700" fill="#f8f9fa"/>
  
  <!-- Title -->
  <text x="600" y="40" text-anchor="middle" font-family="Arial, sans-serif" font-size="24" font-weight="bold" fill="#333">
    Production Deployment Architecture Flow
  </text>
  
  <!-- Input Layer -->
  <g id="input-layer">
    <rect x="50" y="100" width="200" height="100" fill="#e3f2fd" stroke="#2196F3" stroke-width="2" rx="10"/>
    <text x="150" y="130" text-anchor="middle" font-family="Arial" font-size="14" font-weight="bold" fill="#1976D2">
      Data Ingestion
    </text>
    <text x="150" y="150" text-anchor="middle" font-family="Arial" font-size="12" fill="#666">
      • Image Upload API
    </text>
    <text x="150" y="170" text-anchor="middle" font-family="Arial" font-size="12" fill="#666">
      • Batch Processing
    </text>
    <text x="150" y="190" text-anchor="middle" font-family="Arial" font-size="12" fill="#666">
      • Real-time Streaming
    </text>
  </g>
  
  <!-- Preprocessing Layer -->
  <g id="preprocessing-layer">
    <rect x="300" y="100" width="200" height="100" fill="#fff3e0" stroke="#FF9800" stroke-width="2" rx="10"/>
    <text x="400" y="130" text-anchor="middle" font-family="Arial" font-size="14" font-weight="bold" fill="#F57C00">
      Preprocessing
    </text>
    <text x="400" y="150" text-anchor="middle" font-family="Arial" font-size="12" fill="#666">
      • Image Normalization
    </text>
    <text x="400" y="170" text-anchor="middle" font-family="Arial" font-size="12" fill="#666">
      • Bounding Box Extract
    </text>
    <text x="400" y="190" text-anchor="middle" font-family="Arial" font-size="12" fill="#666">
      • Format Conversion
    </text>
  </g>
  
  <!-- Model Inference Layer -->
  <g id="inference-layer">
    <rect x="550" y="100" width="200" height="100" fill="#e8f5e8" stroke="#4CAF50" stroke-width="2" rx="10"/>
    <text x="650" y="130" text-anchor="middle" font-family="Arial" font-size="14" font-weight="bold" fill="#388E3C">
      Model Inference
    </text>
    <text x="650" y="150" text-anchor="middle" font-family="Arial" font-size="12" fill="#666">
      • Binary Classifiers
    </text>
    <text x="650" y="170" text-anchor="middle" font-family="Arial" font-size="12" fill="#666">
      • Ensemble Decision
    </text>
    <text x="650" y="190" text-anchor="middle" font-family="Arial" font-size="12" fill="#666">
      • Confidence Scoring
    </text>
  </g>
  
  <!-- XAI Analysis Layer -->
  <g id="xai-layer">
    <rect x="800" y="100" width="200" height="100" fill="#f3e5f5" stroke="#9C27B0" stroke-width="2" rx="10"/>
    <text x="900" y="130" text-anchor="middle" font-family="Arial" font-size="14" font-weight="bold" fill="#7B1FA2">
      XAI Analysis
    </text>
    <text x="900" y="150" text-anchor="middle" font-family="Arial" font-size="12" fill="#666">
      • Attribution Maps
    </text>
    <text x="900" y="170" text-anchor="middle" font-family="Arial" font-size="12" fill="#666">
      • Feature Importance
    </text>
    <text x="900" y="190" text-anchor="middle" font-family="Arial" font-size="12" fill="#666">
      • Explanation Gen.
    </text>
  </g>
  
  <!-- Storage Layer -->
  <g id="storage-layer">
    <rect x="200" y="300" width="800" height="120" fill="white" stroke="#666" stroke-width="2" rx="10"/>
    <text x="600" y="330" text-anchor="middle" font-family="Arial" font-size="16" font-weight="bold" fill="#333">
      Storage & Database Layer
    </text>
    
    <!-- Model Repository -->
    <ellipse cx="300" cy="380" rx="80" ry="40" fill="#dbeafe" stroke="#3b82f6" stroke-width="2"/>
    <text x="300" y="375" text-anchor="middle" font-family="Arial" font-size="12" font-weight="bold" fill="#1d4ed8">
      Model
    </text>
    <text x="300" y="390" text-anchor="middle" font-family="Arial" font-size="12" font-weight="bold" fill="#1d4ed8">
      Repository
    </text>
    
    <!-- Results Database -->
    <ellipse cx="500" cy="380" rx="80" ry="40" fill="#dcfce7" stroke="#22c55e" stroke-width="2"/>
    <text x="500" y="375" text-anchor="middle" font-family="Arial" font-size="12" font-weight="bold" fill="#15803d">
      Results
    </text>
    <text x="500" y="390" text-anchor="middle" font-family="Arial" font-size="12" font-weight="bold" fill="#15803d">
      Database
    </text>
    
    <!-- Cache Layer -->
    <ellipse cx="700" cy="380" rx="80" ry="40" fill="#fef3c7" stroke="#f59e0b" stroke-width="2"/>
    <text x="700" y="375" text-anchor="middle" font-family="Arial" font-size="12" font-weight="bold" fill="#d97706">
      Cache
    </text>
    <text x="700" y="390" text-anchor="middle" font-family="Arial" font-size="12" font-weight="bold" fill="#d97706">
      Layer
    </text>
    
    <!-- Monitoring Database -->
    <ellipse cx="900" cy="380" rx="80" ry="40" fill="#fce7f3" stroke="#ec4899" stroke-width="2"/>
    <text x="900" y="375" text-anchor="middle" font-family="Arial" font-size="12" font-weight="bold" fill="#be185d">
      Monitoring
    </text>
    <text x="900" y="390" text-anchor="middle" font-family="Arial" font-size="12" font-weight="bold" fill="#be185d">
      Database
    </text>
  </g>
  
  <!-- Monitoring & Analytics Layer -->
  <g id="monitoring-layer">
    <rect x="100" y="500" width="300" height="100" fill="#fef2f2" stroke="#ef4444" stroke-width="2" rx="10"/>
    <text x="250" y="530" text-anchor="middle" font-family="Arial" font-size="14" font-weight="bold" fill="#dc2626">
      Performance Monitoring
    </text>
    <text x="250" y="550" text-anchor="middle" font-family="Arial" font-size="12" fill="#666">
      • Real-time Metrics
    </text>
    <text x="250" y="570" text-anchor="middle" font-family="Arial" font-size="12" fill="#666">
      • Error Rate Tracking
    </text>
    <text x="250" y="590" text-anchor="middle" font-family="Arial" font-size="12" fill="#666">
      • Resource Utilization
    </text>
  </g>
  
  <!-- Output Layer -->
  <g id="output-layer">
    <rect x="700" y="500" width="300" height="100" fill="#f0f9ff" stroke="#0ea5e9" stroke-width="2" rx="10"/>
    <text x="850" y="530" text-anchor="middle" font-family="Arial" font-size="14" font-weight="bold" fill="#0284c7">
      Output & API Response
    </text>
    <text x="850" y="550" text-anchor="middle" font-family="Arial" font-size="12" fill="#666">
      • Classification Results
    </text>
    <text x="850" y="570" text-anchor="middle" font-family="Arial" font-size="12" fill="#666">
      • Attribution Visualizations
    </text>
    <text x="850" y="590" text-anchor="middle" font-family="Arial" font-size="12" fill="#666">
      • Confidence Scores
    </text>
  </g>
  
  <!-- Flow Arrows -->
  <g id="flow-arrows">
    <!-- Horizontal flow -->
    <path d="M 260 150 L 290 150" stroke="#333" stroke-width="3" marker-end="url(#arrowhead)" filter="url(#glow)"/>
    <path d="M 510 150 L 540 150" stroke="#333" stroke-width="3" marker-end="url(#arrowhead)" filter="url(#glow)"/>
    <path d="M 760 150 L 790 150" stroke="#333" stroke-width="3" marker-end="url(#arrowhead)" filter="url(#glow)"/>
    
    <!-- To storage -->
    <path d="M 650 210 L 650 290" stroke="#666" stroke-width="2" marker-end="url(#arrowhead)"/>
    <path d="M 900 210 L 900 290" stroke="#666" stroke-width="2" marker-end="url(#arrowhead)"/>
    
    <!-- From storage to outputs -->
    <path d="M 500 430 L 500 490 L 750 490" stroke="#666" stroke-width="2" marker-end="url(#arrowhead)"/>
    <path d="M 300 430 L 300 490 L 150 490" stroke="#666" stroke-width="2" marker-end="url(#arrowhead)"/>
  </g>
  
  <!-- Performance Indicators -->
  <g id="performance-indicators">
    <circle cx="1050" cy="150" r="30" fill="#22c55e" opacity="0.8"/>
    <text x="1050" y="145" text-anchor="middle" font-family="Arial" font-size="10" font-weight="bold" fill="white">
      99.9%
    </text>
    <text x="1050" y="160" text-anchor="middle" font-family="Arial" font-size="10" font-weight="bold" fill="white">
      Uptime
    </text>
    
    <circle cx="1050" cy="200" r="30" fill="#3b82f6" opacity="0.8"/>
    <text x="1050" y="195" text-anchor="middle" font-family="Arial" font-size="10" font-weight="bold" fill="white">
      &lt;2s
    </text>
    <text x="1050" y="210" text-anchor="middle" font-family="Arial" font-size="10" font-weight="bold" fill="white">
      Latency
    </text>
    
    <circle cx="1050" cy="250" r="30" fill="#f59e0