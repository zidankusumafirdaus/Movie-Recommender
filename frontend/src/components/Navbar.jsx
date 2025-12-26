import React from 'react';
import { Link, useLocation } from 'react-router-dom';

export default function Navbar() {
    const { pathname } = useLocation();

    const linkClass = (p) => `px-4 py-2 rounded-md font-medium text-sm ${pathname === p ? 'bg-primary-700 text-white' : 'text-primary-300 hover:bg-primary-700/30'}`;

    return (
        <nav className="relative z-20">
            <div className="p-5 -mt-6">
                <div className="flex justify-center">
                    <div className="flex gap-5">
                        <Link to="/" className={linkClass('/')} >Dashboard</Link>
                        <Link to="/movies" className={linkClass('/movies')}>All Movies</Link>
                        <Link to="/users" className={linkClass('/users')}>Create User</Link>
                    </div>
                </div>
            </div>
        </nav>
    );
}
