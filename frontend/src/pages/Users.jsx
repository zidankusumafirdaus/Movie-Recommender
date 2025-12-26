import React, { useState } from 'react';
import '../styles/index.css';
import UserForm from '../components/UserForm';
import Navbar from '../components/Navbar';

export default function Users() {
    const [created, setCreated] = useState([]);

    const handleUserCreated = (user) => {
        // keep a short history of created users in local state
        setCreated(prev => [user, ...prev].slice(0, 10));
    };

    return (
        <div className="min-h-screen bg-primary-900 text-white p-6">
            <Navbar />
            <div className="max-w-1xl mx-auto mt-6">
                <section className="grid grid-cols-1 md:grid-cols-[minmax(0,1fr)_30rem] gap-3">
                    <div className="bg-primary-900/30 p-4 rounded-lg">
                        <UserForm onUserCreated={handleUserCreated} />
                    </div>

                    <aside className="bg-primary-900/30 p-4 rounded max-w-xs mx-auto">
                        <h4 className="font-semibold mb-2 text-center text-white">Recently created</h4>
                        {created.length === 0 && <p className="text-primary-300">No users created in this session.</p>}
                        <ul className="mt-3 space-y-2">
                            {created.map((u, i) => (
                                <li key={i} className="bg-primary-900/20 p-3 rounded-lg">
                                    <div className="text-sm text-primary-300">ID: <span className="font-mono">{u.user_id}</span></div>
                                    <div className="text-sm text-gray-200">{u.user_occupation_label || u.user_occupation} • {u.user_gender} • {u.user_age}</div>
                                </li>
                            ))}
                        </ul>
                    </aside>
                </section>
            </div>
        </div>
    );
}
