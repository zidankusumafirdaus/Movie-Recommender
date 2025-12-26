import './styles/index.css';
import React from 'react';
import ReactDOM from 'react-dom/client';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';

ReactDOM.createRoot(document.getElementById('root')).render(
    <React.StrictMode>
        <Router>
            <Routes>
                <Route path="/" element={<App />} />
                <Route path="/movies" element={<Movies />} />
                <Route path="/users" element={<Users />} />
            </Routes>
        </Router>
    </React.StrictMode>
);

import App from './pages/App.jsx';
import Movies from './pages/Movies.jsx';
import Users from './pages/Users.jsx';