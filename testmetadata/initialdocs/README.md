# KMRL Smart Automation - Document Overload Management

A modern, responsive landing page for Kochi Metro Rail Limited's Smart Automation project under SIH 2025. This project addresses Problem Statement 25080 - Document Overload Management through an intuitive, metro-inspired design.

## 🚀 Project Overview

This landing page serves as a knowledge hub for KMRL's document management system, featuring:

- **Work Status Dashboard** - Real-time metrics and progress tracking
- **Document Upload System** - Drag-and-drop file upload with preview
- **Case Studies & Reports** - Organized display of uploaded documents
- **Searchable Catalog** - Advanced filtering and search capabilities
- **Mitigation Strategies** - Expandable content blocks showcasing solutions

## 🎨 Design Features

### Color Palette
- **Primary**: Metro Blue (#0078D4)
- **Secondary**: Leaf Green (#6BBE45)
- **Accent**: Safety Orange (#FF6600)
- **Background**: Light Grey (#F5F7FA)
- **Text**: Charcoal Black (#2E2E2E) and White (#FFFFFF)

### Typography
- **Headings**: Poppins (Bold, modern sans-serif)
- **Body Text**: Roboto (Neutral sans-serif)

### Key Design Elements
- Rounded corners (2xl) with soft shadows
- Grid-based responsive layout
- Metro-inspired visual elements
- Smooth animations and transitions
- Mobile-first responsive design

## 🛠️ Technical Features

### File-Based State Management
- No external database required
- JSON file storage for documents and case studies
- LocalStorage for client-side persistence
- Lightweight and accessible for non-technical users

### Responsive Design
- Mobile-first approach
- Breakpoints: 768px, 480px
- Flexible grid layouts
- Touch-friendly interface

### Interactive Elements
- Drag-and-drop file upload
- Animated counters and progress bars
- Smooth scrolling navigation
- Expandable strategy cards
- Real-time search and filtering

## 📁 Project Structure

```
DataPluse/
├── index.html          # Main HTML file
├── styles.css          # CSS styles and responsive design
├── script.js           # JavaScript functionality
└── README.md           # Project documentation
```

## 🚀 Getting Started

### Prerequisites
- Modern web browser (Chrome, Firefox, Safari, Edge)
- No server setup required - runs locally

### Installation
1. Clone or download the project files
2. Open `index.html` in your web browser
3. The application will load with sample data

### Usage
1. **Navigation**: Use the top navigation to move between sections
2. **Upload Files**: Drag and drop files or click to select
3. **Search**: Use the search box to find specific documents
4. **Filter**: Use category and tag filters to narrow results
5. **View Details**: Click on document cards to view details

## 📊 Features Breakdown

### Hero Section
- Problem statement display
- Call-to-action buttons
- Animated metro line visualization
- Responsive design

### Work Status Dashboard
- Documents processed counter
- Mitigation strategies implemented
- Pending work queue
- Bilingual support accuracy
- Animated progress bars

### Upload & Case Studies
- Drag-and-drop file upload
- File type validation (PDF, DOC, DOCX, TXT, MD)
- Automatic categorization
- Tag generation
- Case study display

### Mitigation Strategies
- Expandable accordion cards
- Strategy details and metrics
- Smooth animations
- Professional presentation

### Document Catalog
- Search functionality
- Category filtering
- Tag-based filtering
- Card-based layout
- File metadata display

### About Team
- Team information
- Project statistics
- Feature highlights
- Professional presentation

## 🔧 Customization

### Adding New Document Types
1. Update the `isValidFile()` function in `script.js`
2. Add new MIME types to the validation array
3. Update the `getFileType()` function for display

### Modifying Color Scheme
1. Update CSS custom properties in `:root` selector
2. Colors are defined as CSS variables for easy modification

### Adding New Categories
1. Update the `determineCategory()` function
2. Add new options to the category filter dropdown
3. Update the category mapping logic

## 📱 Browser Support

- Chrome 60+
- Firefox 55+
- Safari 12+
- Edge 79+

## 🎯 Key Benefits Highlighted

1. **Reduced Information Latency** - Faster document processing
2. **Compliance Automation** - Automated compliance checking
3. **Centralized Logs** - Unified logging and audit trail
4. **Bilingual Support** - English and Malayalam document handling
5. **Optimized OCR** - Advanced optical character recognition
6. **Persona-Driven Alerts** - Intelligent notification system

## 👥 Team Information

**Team Data Pulse (ISIH062)**
- SIH 2025 Project
- Smart Automation for KMRL
- Focus on user-centered design and innovation

## 📄 License

This project is developed for SIH 2025 and Kochi Metro Rail Limited.

## 🤝 Contributing

This is a project-specific implementation for SIH 2025. For modifications or improvements, please contact Team Data Pulse.

## 📞 Support

For technical support or questions about the implementation, please refer to the team documentation or contact the development team.

---

**Note**: This is a client-side application that uses localStorage for data persistence. No server setup or external dependencies are required for basic functionality.
