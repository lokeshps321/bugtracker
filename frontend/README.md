# BugFlow Frontend

A professional, interactive React-based frontend for the BugFlow AI-powered bug management system.

## Features

- **User Authentication**: Secure login with JWT tokens
- **Bug Reporting**: Report bugs with AI-powered severity and team prediction
- **Bug Management**: View, update, and track bug status
- **Analytics Dashboard**: Real-time insights and feedback tracking
- **Responsive Design**: Mobile-friendly interface with TailwindCSS
- **Modern UI**: Built with React, TypeScript, and Lucide icons

## Tech Stack

- **React 19** - UI framework
- **TypeScript** - Type safety
- **TailwindCSS** - Styling
- **Lucide React** - Icons
- **Axios** - HTTP client
- **React Router** - Navigation

## Getting Started

### Prerequisites

- Node.js 14+ and npm

### Installation

1. Install dependencies:
```bash
npm install
```

2. Configure environment variables in `.env`:
```
REACT_APP_API_URL=http://localhost:8000
```

### Running the App

Start the development server:
```bash
npm start
```

The app will open at [http://localhost:3000](http://localhost:3000)

### Demo Credentials

- **Project Manager**: pm1@example.com / password
- **Tester**: tester1@example.com / password
- **Developer**: dev1@example.com / password

## Project Structure

```
src/
├── components/          # React components
│   ├── Login.tsx       # Login page
│   ├── Dashboard.tsx   # Main dashboard
│   ├── BugReportForm.tsx # Bug reporting
│   ├── BugList.tsx     # Bug list view
│   ├── Analytics.tsx   # Analytics dashboard
│   └── ProtectedRoute.tsx # Route protection
├── context/            # React context
│   └── AuthContext.tsx # Authentication state
├── services/           # API services
│   └── api.ts         # API client
└── App.tsx            # Main app component
```

## Building for Production

```bash
npm run build
```

This creates an optimized production build in the `build/` folder.

## API Integration

The frontend connects to the BugFlow backend API at `http://localhost:8000`. Ensure the backend is running before starting the frontend.

### Key API Endpoints

- `POST /token` - User authentication
- `POST /report_bug` - Report a new bug
- `GET /bugs` - Fetch all bugs
- `POST /update_bug` - Update bug status
- `POST /predict` - Get AI predictions for bug severity and team
- `GET /feedback_count` - Get feedback statistics
- `GET /feedback_history` - Get feedback trend data
- `GET /notifications` - Get system notifications
