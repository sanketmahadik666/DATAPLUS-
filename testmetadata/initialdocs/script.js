// Global state management
let documents = [];
let caseStudies = [];
let currentFilter = {
    category: '',
    tag: '',
    search: ''
};

// Initialize the application
document.addEventListener('DOMContentLoaded', function() {
    initializeApp();
    setupEventListeners();
    loadData();
    animateCounters();
    setupScrollAnimations();
});

// Initialize application
function initializeApp() {
    // Load existing data from localStorage
    const savedDocuments = localStorage.getItem('kmrl_documents');
    const savedCaseStudies = localStorage.getItem('kmrl_case_studies');
    
    if (savedDocuments) {
        documents = JSON.parse(savedDocuments);
    }
    
    if (savedCaseStudies) {
        caseStudies = JSON.parse(savedCaseStudies);
    }
    
    // Add some sample data if none exists
    if (documents.length === 0) {
        addSampleData();
    }
    
    renderCatalog();
    renderCaseStudies();
}

// Setup event listeners
function setupEventListeners() {
    // Mobile navigation
    const hamburger = document.querySelector('.hamburger');
    const navMenu = document.querySelector('.nav-menu');
    
    if (hamburger && navMenu) {
        hamburger.addEventListener('click', function() {
            navMenu.classList.toggle('active');
        });
    }
    
    // Smooth scrolling for navigation links
    const navLinks = document.querySelectorAll('.nav-link');
    navLinks.forEach(link => {
        link.addEventListener('click', function(e) {
            e.preventDefault();
            const targetId = this.getAttribute('href').substring(1);
            scrollToSection(targetId);
            
            // Update active nav link
            navLinks.forEach(l => l.classList.remove('active'));
            this.classList.add('active');
            
            // Close mobile menu
            if (navMenu) {
                navMenu.classList.remove('active');
            }
        });
    });
    
    // File upload
    const uploadZone = document.getElementById('uploadZone');
    const fileInput = document.getElementById('fileInput');
    
    if (uploadZone && fileInput) {
        // Drag and drop events
        uploadZone.addEventListener('dragover', handleDragOver);
        uploadZone.addEventListener('dragleave', handleDragLeave);
        uploadZone.addEventListener('drop', handleDrop);
        uploadZone.addEventListener('click', () => fileInput.click());
        
        // File input change
        fileInput.addEventListener('change', handleFileSelect);
    }
    
    // Search and filter
    const searchInput = document.getElementById('searchInput');
    const categoryFilter = document.getElementById('categoryFilter');
    const tagFilter = document.getElementById('tagFilter');
    
    if (searchInput) {
        searchInput.addEventListener('input', handleSearch);
    }
    
    if (categoryFilter) {
        categoryFilter.addEventListener('change', handleFilter);
    }
    
    if (tagFilter) {
        tagFilter.addEventListener('change', handleFilter);
    }
}

// Smooth scroll to section
function scrollToSection(sectionId) {
    const section = document.getElementById(sectionId);
    if (section) {
        const offsetTop = section.offsetTop - 70; // Account for fixed navbar
        window.scrollTo({
            top: offsetTop,
            behavior: 'smooth'
        });
    }
}

// Animate counters
function animateCounters() {
    const counters = document.querySelectorAll('.status-number');
    
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                const counter = entry.target;
                const target = parseInt(counter.getAttribute('data-target'));
                animateCounter(counter, target);
                observer.unobserve(counter);
            }
        });
    });
    
    counters.forEach(counter => observer.observe(counter));
}

function animateCounter(element, target) {
    let current = 0;
    const increment = target / 100;
    const timer = setInterval(() => {
        current += increment;
        if (current >= target) {
            current = target;
            clearInterval(timer);
        }
        element.textContent = Math.floor(current).toLocaleString();
    }, 20);
}

// Setup scroll animations
function setupScrollAnimations() {
    const animatedElements = document.querySelectorAll('.status-card, .benefit-item, .case-study-card, .document-card');
    
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('fade-in', 'visible');
            }
        });
    }, { threshold: 0.1 });
    
    animatedElements.forEach(element => {
        element.classList.add('fade-in');
        observer.observe(element);
    });
}

// File upload handlers
function handleDragOver(e) {
    e.preventDefault();
    e.currentTarget.classList.add('dragover');
}

function handleDragLeave(e) {
    e.currentTarget.classList.remove('dragover');
}

function handleDrop(e) {
    e.preventDefault();
    e.currentTarget.classList.remove('dragover');
    
    const files = Array.from(e.dataTransfer.files);
    processFiles(files);
}

function handleFileSelect(e) {
    const files = Array.from(e.target.files);
    processFiles(files);
}

function processFiles(files) {
    files.forEach(file => {
        if (isValidFile(file)) {
            const document = createDocumentFromFile(file);
            documents.push(document);
            saveData();
            renderCatalog();
            showMessage('File uploaded successfully!', 'success');
        } else {
            showMessage('Invalid file type. Please upload PDF, DOC, DOCX, TXT, or MD files.', 'error');
        }
    });
}

function isValidFile(file) {
    const validTypes = [
        'application/pdf',
        'application/msword',
        'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
        'text/plain',
        'text/markdown'
    ];
    return validTypes.includes(file.type);
}

function createDocumentFromFile(file) {
    return {
        id: Date.now() + Math.random(),
        title: file.name.replace(/\.[^/.]+$/, ""),
        type: getFileType(file.type),
        size: file.size,
        uploadDate: new Date().toISOString(),
        tags: generateTags(file.name),
        category: determineCategory(file.name),
        content: file.name // In a real app, you'd process the file content
    };
}

function getFileType(mimeType) {
    const typeMap = {
        'application/pdf': 'PDF',
        'application/msword': 'DOC',
        'application/vnd.openxmlformats-officedocument.wordprocessingml.document': 'DOCX',
        'text/plain': 'TXT',
        'text/markdown': 'MD'
    };
    return typeMap[mimeType] || 'Unknown';
}

function generateTags(filename) {
    const tags = [];
    const lowerName = filename.toLowerCase();
    
    if (lowerName.includes('urgent') || lowerName.includes('priority')) {
        tags.push('urgent');
    }
    if (lowerName.includes('review') || lowerName.includes('draft')) {
        tags.push('review');
    }
    if (lowerName.includes('final') || lowerName.includes('approved')) {
        tags.push('completed');
    }
    
    return tags;
}

function determineCategory(filename) {
    const lowerName = filename.toLowerCase();
    
    if (lowerName.includes('report') || lowerName.includes('analysis')) {
        return 'reports';
    }
    if (lowerName.includes('strategy') || lowerName.includes('mitigation')) {
        return 'strategies';
    }
    if (lowerName.includes('note') || lowerName.includes('brainstorm')) {
        return 'notes';
    }
    if (lowerName.includes('compliance') || lowerName.includes('audit')) {
        return 'compliance';
    }
    
    return 'reports'; // Default category
}

// Search and filter handlers
function handleSearch(e) {
    currentFilter.search = e.target.value.toLowerCase();
    renderCatalog();
}

function handleFilter(e) {
    if (e.target.id === 'categoryFilter') {
        currentFilter.category = e.target.value;
    } else if (e.target.id === 'tagFilter') {
        currentFilter.tag = e.target.value;
    }
    renderCatalog();
}

// Render functions
function renderCatalog() {
    const catalogGrid = document.getElementById('catalogGrid');
    if (!catalogGrid) return;
    
    const filteredDocuments = documents.filter(doc => {
        const matchesSearch = !currentFilter.search || 
            doc.title.toLowerCase().includes(currentFilter.search) ||
            doc.type.toLowerCase().includes(currentFilter.search);
        
        const matchesCategory = !currentFilter.category || 
            doc.category === currentFilter.category;
        
        const matchesTag = !currentFilter.tag || 
            doc.tags.includes(currentFilter.tag);
        
        return matchesSearch && matchesCategory && matchesTag;
    });
    
    catalogGrid.innerHTML = filteredDocuments.map(doc => `
        <div class="document-card" onclick="viewDocument('${doc.id}')">
            <div class="document-header">
                <div>
                    <div class="document-title">${doc.title}</div>
                    <div class="document-meta">
                        ${formatFileSize(doc.size)} • ${formatDate(doc.uploadDate)}
                    </div>
                </div>
                <div class="document-type">${doc.type}</div>
            </div>
            <div class="document-tags">
                ${doc.tags.map(tag => `<span class="tag">${tag}</span>`).join('')}
            </div>
        </div>
    `).join('');
}

function renderCaseStudies() {
    const caseStudiesGrid = document.getElementById('caseStudiesGrid');
    if (!caseStudiesGrid) return;
    
    // Generate case studies from documents
    const caseStudyDocs = documents.filter(doc => 
        doc.category === 'strategies' || doc.category === 'reports'
    ).slice(0, 6);
    
    caseStudiesGrid.innerHTML = caseStudyDocs.map(doc => `
        <div class="case-study-card">
            <div class="case-study-header">
                <div class="case-study-title">${doc.title}</div>
                <div class="case-study-date">${formatDate(doc.uploadDate)}</div>
            </div>
            <div class="case-study-content">
                ${generateCaseStudyContent(doc)}
            </div>
            <div class="case-study-tags">
                ${doc.tags.map(tag => `<span class="tag">${tag}</span>`).join('')}
            </div>
        </div>
    `).join('');
}

function generateCaseStudyContent(doc) {
    const templates = [
        `This document outlines key strategies for improving ${doc.category} processes within the KMRL framework.`,
        `Analysis of current ${doc.category} methodologies and proposed improvements for enhanced efficiency.`,
        `Comprehensive review of ${doc.category} best practices and implementation guidelines.`
    ];
    
    return templates[Math.floor(Math.random() * templates.length)];
}

// Utility functions
function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

function formatDate(dateString) {
    const date = new Date(dateString);
    return date.toLocaleDateString('en-US', {
        year: 'numeric',
        month: 'short',
        day: 'numeric'
    });
}

function viewDocument(id) {
    const doc = documents.find(d => d.id == id);
    if (doc) {
        alert(`Viewing document: ${doc.title}\nType: ${doc.type}\nSize: ${formatFileSize(doc.size)}\nUploaded: ${formatDate(doc.uploadDate)}`);
    }
}

function toggleStrategy(element) {
    const card = element.parentElement;
    card.classList.toggle('active');
}

function showMessage(text, type) {
    // Remove existing messages
    const existingMessages = document.querySelectorAll('.message');
    existingMessages.forEach(msg => msg.remove());
    
    // Create new message
    const message = document.createElement('div');
    message.className = `message ${type}`;
    message.textContent = text;
    
    // Insert at the top of the upload section
    const uploadSection = document.getElementById('upload');
    if (uploadSection) {
        uploadSection.insertBefore(message, uploadSection.firstChild);
        
        // Auto remove after 5 seconds
        setTimeout(() => {
            message.remove();
        }, 5000);
    }
}

// Data persistence
function saveData() {
    localStorage.setItem('kmrl_documents', JSON.stringify(documents));
    localStorage.setItem('kmrl_case_studies', JSON.stringify(caseStudies));
}

function loadData() {
    const savedDocuments = localStorage.getItem('kmrl_documents');
    const savedCaseStudies = localStorage.getItem('kmrl_case_studies');
    
    if (savedDocuments) {
        documents = JSON.parse(savedDocuments);
    }
    
    if (savedCaseStudies) {
        caseStudies = JSON.parse(savedCaseStudies);
    }
}

// Add sample data
function addSampleData() {
    const sampleDocuments = [
        {
            id: 1,
            title: "Document Processing Optimization Report",
            type: "PDF",
            size: 2048576,
            uploadDate: new Date(Date.now() - 86400000).toISOString(),
            tags: ["completed", "optimization"],
            category: "reports",
            content: "Sample report content"
        },
        {
            id: 2,
            title: "Mitigation Strategy for Compliance Issues",
            type: "DOCX",
            size: 1024000,
            uploadDate: new Date(Date.now() - 172800000).toISOString(),
            tags: ["urgent", "compliance"],
            category: "strategies",
            content: "Sample strategy content"
        },
        {
            id: 3,
            title: "Brainstorming Notes - OCR Enhancement",
            type: "TXT",
            size: 512000,
            uploadDate: new Date(Date.now() - 259200000).toISOString(),
            tags: ["review", "innovation"],
            category: "notes",
            content: "Sample notes content"
        },
        {
            id: 4,
            title: "Compliance Audit Results Q4 2024",
            type: "PDF",
            size: 3072000,
            uploadDate: new Date(Date.now() - 345600000).toISOString(),
            tags: ["completed", "audit"],
            category: "compliance",
            content: "Sample audit content"
        },
        {
            id: 5,
            title: "Bilingual Support Implementation Plan",
            type: "DOCX",
            size: 1536000,
            uploadDate: new Date(Date.now() - 432000000).toISOString(),
            tags: ["urgent", "implementation"],
            category: "strategies",
            content: "Sample implementation content"
        },
        {
            id: 6,
            title: "System Performance Analysis",
            type: "PDF",
            size: 2560000,
            uploadDate: new Date(Date.now() - 518400000).toISOString(),
            tags: ["review", "performance"],
            category: "reports",
            content: "Sample analysis content"
        }
    ];
    
    documents = sampleDocuments;
    saveData();
}

// Update progress bars based on data
function updateProgressBars() {
    const progressBars = document.querySelectorAll('.progress-bar');
    
    progressBars.forEach(bar => {
        const width = bar.getAttribute('data-width');
        setTimeout(() => {
            bar.style.width = width + '%';
        }, 500);
    });
}

// Initialize progress bars after page load
window.addEventListener('load', function() {
    updateProgressBars();
});

// Export functions for global access
window.scrollToSection = scrollToSection;
window.toggleStrategy = toggleStrategy;
