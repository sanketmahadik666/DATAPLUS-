# KMRL Smart Automation - Style Guide

## 🎨 Visual Identity

### Brand Colors

#### Primary Colors
- **Metro Blue**: `#0078D4`
  - Usage: Primary buttons, links, icons, progress bars
  - RGB: (0, 120, 212)
  - HSL: (207, 100%, 42%)

- **Leaf Green**: `#6BBE45`
  - Usage: Secondary elements, success states, completion indicators
  - RGB: (107, 190, 69)
  - HSL: (100, 50%, 51%)

- **Safety Orange**: `#FF6600`
  - Usage: Accent elements, CTAs, alerts, highlights
  - RGB: (255, 102, 0)
  - HSL: (24, 100%, 50%)

#### Neutral Colors
- **Charcoal Black**: `#2E2E2E`
  - Usage: Primary text, headings
  - RGB: (46, 46, 46)
  - HSL: (0, 0%, 18%)

- **White**: `#FFFFFF`
  - Usage: Backgrounds, contrast text
  - RGB: (255, 255, 255)
  - HSL: (0, 0%, 100%)

- **Light Grey**: `#F5F7FA`
  - Usage: Background sections, card backgrounds
  - RGB: (245, 247, 250)
  - HSL: (220, 20%, 97%)

- **Border Grey**: `#E8E8E8`
  - Usage: Borders, dividers, subtle elements
  - RGB: (232, 232, 232)
  - HSL: (0, 0%, 91%)

### Typography

#### Font Families
- **Primary (Headings)**: Poppins
  - Weights: 300, 400, 500, 600, 700
  - Usage: All headings, titles, and emphasis text

- **Secondary (Body)**: Roboto
  - Weights: 300, 400, 500
  - Usage: Body text, descriptions, labels

#### Font Sizes
```css
/* Headings */
.hero-title: 3rem (48px)
.section-title: 2.5rem (40px)
.benefits-title: 2rem (32px)
.card-title: 1.2rem (19.2px)

/* Body Text */
.hero-description: 1.1rem (17.6px)
.body-text: 1rem (16px)
.small-text: 0.9rem (14.4px)
.caption: 0.8rem (12.8px)
```

#### Line Heights
- Headings: 1.2
- Body text: 1.6
- Descriptions: 1.7

### Spacing System

#### Padding & Margins
```css
/* Small spacing */
.padding-sm: 0.5rem (8px)
.margin-sm: 0.5rem (8px)

/* Medium spacing */
.padding-md: 1rem (16px)
.margin-md: 1rem (16px)

/* Large spacing */
.padding-lg: 2rem (32px)
.margin-lg: 2rem (32px)

/* Extra large spacing */
.padding-xl: 3rem (48px)
.margin-xl: 3rem (48px)

/* Section spacing */
.section-padding: 80px 0
.container-padding: 0 1rem
```

### Border Radius
```css
/* Small radius */
.border-radius-sm: 0.5rem (8px)

/* Medium radius */
.border-radius-md: 1rem (16px)

/* Large radius */
.border-radius-lg: 1.5rem (24px)

/* Extra large radius */
.border-radius-xl: 2rem (32px)
```

### Shadows
```css
/* Light shadow */
.shadow: 0 4px 6px rgba(0, 0, 0, 0.1)

/* Medium shadow */
.shadow-md: 0 6px 12px rgba(0, 0, 0, 0.15)

/* Large shadow */
.shadow-lg: 0 10px 25px rgba(0, 0, 0, 0.15)

/* Card shadow */
.card-shadow: 0 4px 6px rgba(0, 0, 0, 0.1)
```

## 🎯 Component Styles

### Buttons

#### Primary Button
```css
.btn-primary {
    background: var(--accent-color);
    color: var(--white);
    padding: 1rem 2rem;
    border-radius: 1rem;
    font-family: 'Poppins', sans-serif;
    font-weight: 500;
    border: none;
    cursor: pointer;
    transition: all 0.3s ease;
}

.btn-primary:hover {
    background: #e55a00;
    transform: translateY(-2px);
    box-shadow: var(--shadow-lg);
}
```

#### Secondary Button
```css
.btn-secondary {
    background: transparent;
    color: var(--white);
    border: 2px solid var(--white);
    padding: 1rem 2rem;
    border-radius: 1rem;
    font-family: 'Poppins', sans-serif;
    font-weight: 500;
    cursor: pointer;
    transition: all 0.3s ease;
}

.btn-secondary:hover {
    background: var(--white);
    color: var(--primary-color);
    transform: translateY(-2px);
}
```

### Cards

#### Status Card
```css
.status-card {
    background: var(--white);
    border-radius: 1rem;
    padding: 2rem;
    box-shadow: var(--shadow);
    border: 1px solid var(--light-gray);
    transition: transform 0.3s ease, box-shadow 0.3s ease;
    display: flex;
    align-items: center;
    gap: 1.5rem;
}

.status-card:hover {
    transform: translateY(-5px);
    box-shadow: var(--shadow-lg);
}
```

#### Document Card
```css
.document-card {
    background: var(--white);
    border-radius: 1rem;
    padding: 1.5rem;
    box-shadow: var(--shadow);
    border: 1px solid var(--light-gray);
    transition: transform 0.3s ease;
    cursor: pointer;
}

.document-card:hover {
    transform: translateY(-5px);
    box-shadow: var(--shadow-lg);
}
```

### Form Elements

#### Input Fields
```css
.input-field {
    width: 100%;
    padding: 1rem;
    border: 2px solid var(--light-gray);
    border-radius: 1rem;
    font-size: 1rem;
    transition: border-color 0.3s ease;
}

.input-field:focus {
    outline: none;
    border-color: var(--primary-color);
}
```

#### Select Dropdowns
```css
.select-field {
    padding: 1rem;
    border: 2px solid var(--light-gray);
    border-radius: 1rem;
    font-size: 1rem;
    background: var(--white);
    cursor: pointer;
    transition: border-color 0.3s ease;
}

.select-field:focus {
    outline: none;
    border-color: var(--primary-color);
}
```

### Tags
```css
.tag {
    background: var(--primary-color);
    color: var(--white);
    padding: 0.25rem 0.75rem;
    border-radius: 1rem;
    font-size: 0.8rem;
    font-weight: 500;
    display: inline-block;
}

.tag.urgent {
    background: var(--accent-color);
}

.tag.completed {
    background: var(--secondary-color);
}
```

## 📱 Responsive Design

### Breakpoints
```css
/* Mobile */
@media (max-width: 480px) {
    .hero-title { font-size: 2rem; }
    .section-title { font-size: 2rem; }
}

/* Tablet */
@media (max-width: 768px) {
    .hero-container { grid-template-columns: 1fr; }
    .status-grid { grid-template-columns: 1fr; }
    .team-content { grid-template-columns: 1fr; }
}

/* Desktop */
@media (min-width: 769px) {
    .hero-container { grid-template-columns: 1fr 1fr; }
    .status-grid { grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); }
}
```

### Grid Systems
```css
/* Status Grid */
.status-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
    gap: 2rem;
}

/* Benefits Grid */
.benefits-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
    gap: 2rem;
}

/* Catalog Grid */
.catalog-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
    gap: 2rem;
}
```

## 🎭 Animations

### Fade In Animation
```css
.fade-in {
    opacity: 0;
    transform: translateY(30px);
    transition: all 0.6s ease;
}

.fade-in.visible {
    opacity: 1;
    transform: translateY(0);
}
```

### Slide Animations
```css
.slide-in-left {
    opacity: 0;
    transform: translateX(-50px);
    transition: all 0.6s ease;
}

.slide-in-right {
    opacity: 0;
    transform: translateX(50px);
    transition: all 0.6s ease;
}
```

### Hover Effects
```css
.hover-lift:hover {
    transform: translateY(-5px);
    box-shadow: var(--shadow-lg);
}

.hover-scale:hover {
    transform: scale(1.02);
}
```

## 🎨 Metro-Inspired Elements

### Metro Line Animation
```css
.metro-line {
    width: 100%;
    height: 4px;
    background: var(--white);
    border-radius: 2px;
    position: relative;
}

.metro-line::after {
    content: '';
    position: absolute;
    top: 50%;
    left: 0;
    width: 20%;
    height: 4px;
    background: var(--accent-color);
    border-radius: 2px;
    animation: metroMove 3s ease-in-out infinite;
}

@keyframes metroMove {
    0%, 100% { left: 0; }
    50% { left: 80%; }
}
```

### Progress Bars
```css
.progress-bar {
    height: 100%;
    background: linear-gradient(90deg, var(--primary-color), var(--secondary-color));
    border-radius: 3px;
    transition: width 2s ease;
}
```

## 📐 Layout Guidelines

### Container Widths
- **Max Width**: 1200px
- **Padding**: 0 1rem (mobile), 0 2rem (desktop)
- **Margin**: 0 auto (centered)

### Section Spacing
- **Vertical Padding**: 80px 0
- **Section Gaps**: 4rem between major sections
- **Card Gaps**: 2rem between cards

### Content Hierarchy
1. **Hero Section** - Full viewport height
2. **Status Dashboard** - Key metrics display
3. **Upload Section** - File management
4. **Mitigation Strategies** - Expandable content
5. **Catalog** - Searchable document library
6. **About Team** - Project information
7. **Footer** - Contact and links

## 🎯 Accessibility

### Color Contrast
- All text meets WCAG AA standards
- Minimum contrast ratio of 4.5:1 for normal text
- Minimum contrast ratio of 3:1 for large text

### Focus States
```css
.focusable:focus {
    outline: 2px solid var(--primary-color);
    outline-offset: 2px;
}
```

### Screen Reader Support
- Semantic HTML structure
- ARIA labels for interactive elements
- Alt text for images and icons

## 📋 Usage Guidelines

### Do's
- Use the defined color palette consistently
- Maintain proper spacing and typography hierarchy
- Ensure all interactive elements have hover states
- Test on multiple devices and screen sizes
- Keep animations subtle and purposeful

### Don'ts
- Don't use colors outside the defined palette
- Don't mix different font families
- Don't create overly complex animations
- Don't ignore responsive design principles
- Don't compromise accessibility for aesthetics

---

This style guide ensures consistency and professionalism across the KMRL Smart Automation landing page while maintaining the metro-inspired aesthetic and user-friendly design principles.
