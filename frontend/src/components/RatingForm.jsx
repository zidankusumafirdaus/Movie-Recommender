import React, { useState } from 'react';
import { postRating } from '../api/api';

export default function RatingForm({ defaultUserId, onSuccess }) {
    const [userId, setUserId] = useState(() => defaultUserId || '');
    const [movieId, setMovieId] = useState('');
    const [rating, setRating] = useState('');
    const [status, setStatus] = useState(null);

    const handleSubmit = async (e) => {
        e.preventDefault();
        setStatus('sending');
        try {
            await postRating({
                user_id: String(userId),
                movie_id: String(movieId),
                user_rating: Number(rating),
            });
            setStatus('ok');
            setMovieId('');
            setRating('');
            if (onSuccess) onSuccess();
            // clear status after short delay
            setTimeout(() => setStatus(null), 2000);
        } catch (err) {
            console.error(err);
            setStatus('error');
        }
    };

    return (
        <form onSubmit={handleSubmit} className="bg-primary-700/30 p-3 rounded-md border border-primary-700">
            <h4 className="font-semibold mb-7 text-white">Submit a rating</h4>

            <div className="space-y-3">
                <div>
                    <label className="text-xs text-gray-300">User ID</label>
                    <input value={userId} onChange={e => setUserId(e.target.value)} placeholder="User ID" className="w-full mt-1 p-2 rounded bg-white/5 focus:outline-none focus:ring-2 focus:ring-primary-500" />
                </div>

                <div>
                    <label className="text-xs text-gray-300">Movie ID</label>
                    <input value={movieId} onChange={e => setMovieId(e.target.value)} placeholder="Movie ID" className="w-full mt-1 p-2 rounded bg-white/5 focus:outline-none focus:ring-2 focus:ring-primary-500" />
                </div>

                <div>
                    <label className="text-xs text-gray-300">Rating</label>
                    <input value={rating} onChange={e => setRating(e.target.value)} placeholder="Rating (e.g. 4.0)" className="w-full mt-1 p-2 rounded bg-white/5 focus:outline-none focus:ring-2 focus:ring-primary-500" />
                </div>

                <div className="flex items-center gap-2">
                    <button type="submit" className="flex-1 bg-primary-500 hover:bg-primary-700 text-white py-2 px-3 rounded-md">Send Rating</button>
                </div>

                <div>
                    {status === 'sending' && <span className="text-primary-300">Sending...</span>}
                    {status === 'ok' && <span className="text-green-300">Sent ✓</span>}
                    {status === 'error' && <span className="text-red-300">Error</span>}
                </div>
            </div>
        </form>
    );
}