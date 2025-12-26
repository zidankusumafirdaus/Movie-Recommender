import React, { useState } from 'react';
import RatingForm from './RatingForm';
import RatedMovieCard from './RatedMovieCard';
import { getRatedMovies } from '../api/api';

export default function RatingsSidebar({ initialUserId = '100' }) {
    const [userId, setUserId] = useState(initialUserId);
    const [rated, setRated] = useState([]);
    const [loadingRated, setLoadingRated] = useState(false);

    const fetchRated = async () => {
        setLoadingRated(true);
        try {
            const res = await getRatedMovies(userId);
            setRated(res.data.rated_movies || []);
        } catch (err) {
            console.error(err);
            setRated([]);
        } finally {
            setLoadingRated(false);
        }
    };

    return (
        <aside className="bg-primary-900/40 p-4 rounded space-y-4">
            <div>
                <RatingForm defaultUserId={userId} onSuccess={fetchRated} />
            </div>

            <div className="bg-primary-700/30 p-3 rounded-md border border-primary-700">
                <h4 className="font-semibold mb-7 text-white">History Rated</h4>

                <label className="block text-sm text-gray-200 mb-1">User ID</label>
                <input value={userId} onChange={e => setUserId(e.target.value)} className="w-full p-2 rounded bg-white/5 focus:outline-none focus:ring-2 focus:ring-primary-500" />

                <div className="flex gap-2 mb-3 mt-3">
                    <button onClick={fetchRated} className="bg-primary-500 hover:bg-primary-700 text-white py-2 px-3 rounded">Load Rated</button>
                    {loadingRated && <span className="text-primary-300">Loading...</span>}
                </div>

                <div className="space-y-3">
                    {rated.length === 0 && <p className="text-primary-300">No rated movies.</p>}
                    {rated.map((m) => (
                        <RatedMovieCard key={m.movie_id} movie={m} />
                    ))}
                </div>
            </div>
        </aside>
    );
}
