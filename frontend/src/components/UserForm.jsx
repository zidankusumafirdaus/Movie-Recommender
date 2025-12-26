import React, { useState } from 'react';
import { postUser } from '../api/api';

export default function UserForm({ onUserCreated }) {
    const [userId, setUserId] = useState('');
    const [age, setAge] = useState('');
    const [gender, setGender] = useState('');
    const [occupation, setOccupation] = useState('');
    const [occupationLabel, setOccupationLabel] = useState('');
    const [zip, setZip] = useState('');
    const [status, setStatus] = useState(null);

    const handleSubmit = async (e) => {
        e.preventDefault();
        setStatus('sending');
        try {
            const payload = {
                user_id: String(userId),
                user_age: Number(age),
                user_gender: String(gender),
                user_occupation: String(occupation),
                user_occupation_label: String(occupationLabel),
                user_zip_code: String(zip),
            };
            const res = await postUser(payload);
            setStatus('ok');
            // notify parent with returned data (if any)
            if (onUserCreated) onUserCreated(res.data || payload);
            // clear form
            setUserId(''); setAge(''); setGender(''); setOccupation(''); setOccupationLabel(''); setZip('');
        } catch (err) {
            console.error(err);
            setStatus('error');
        }
    };

    return (
        <form onSubmit={handleSubmit} className="bg-primary-900/40 p-4 rounded-md shadow-inner">
            <h4 className="font-semibold mb-3 text-white">Create User</h4>

            <div className="space-y-3">
                <div>
                    <label className="text-xs text-gray-300">User ID</label>
                    <input value={userId} onChange={e => setUserId(e.target.value)} placeholder="User ID" className="w-full mt-1 p-2 rounded bg-white/5 focus:outline-none focus:ring-2 focus:ring-primary-500" />
                </div>

                <div className="grid grid-cols-2 gap-2">
                    <div>
                        <label className="text-xs text-gray-300">Age</label>
                        <input value={age} onChange={e => setAge(e.target.value)} placeholder="Age" className="w-full mt-1 p-2 rounded bg-white/5 focus:outline-none focus:ring-2 focus:ring-primary-500" />
                    </div>
                    <div>
                        <label className="text-xs text-gray-300">Gender</label>
                        <input value={gender} onChange={e => setGender(e.target.value)} placeholder="Gender (M/F)" className="w-full mt-1 p-2 rounded bg-white/5 focus:outline-none focus:ring-2 focus:ring-primary-500" />
                    </div>
                </div>

                <div>
                    <label className="text-xs text-gray-300">Occupation code</label>
                    <input value={occupation} onChange={e => setOccupation(e.target.value)} placeholder="Occupation (code)" className="w-full mt-1 p-2 rounded bg-white/5 focus:outline-none focus:ring-2 focus:ring-primary-500" />
                </div>

                <div>
                    <label className="text-xs text-gray-300">Occupation label</label>
                    <input value={occupationLabel} onChange={e => setOccupationLabel(e.target.value)} placeholder="Occupation label" className="w-full mt-1 p-2 rounded bg-white/5 focus:outline-none focus:ring-2 focus:ring-primary-500" />
                </div>

                <div>
                    <label className="text-xs text-gray-300">Zip code</label>
                    <input value={zip} onChange={e => setZip(e.target.value)} placeholder="Zip code" className="w-full mt-1 p-2 rounded bg-white/5 focus:outline-none focus:ring-2 focus:ring-primary-500" />
                </div>

                <div className="flex items-center gap-2">
                    <button type="submit" className="flex-1 bg-primary-500 hover:bg-primary-700 text-white py-2 px-3 rounded-md">Create User</button>
                </div>

                <div>
                    {status === 'sending' && <span className="text-primary-300">Sending...</span>}
                    {status === 'ok' && <span className="text-green-300">Created ✓</span>}
                    {status === 'error' && <span className="text-red-300">Error</span>}
                </div>
            </div>
        </form>
    );
}
