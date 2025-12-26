import React from 'react';
import MovieCard from './MovieCard';

export default function RecommendationList({ recommendations }) {
    if (!recommendations || recommendations.length === 0) {
        return <p className="text-center text-gray-300">No recommendations yet.</p>;
    }

    return (
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-2 gap-4">
            {recommendations.map((r, idx) => (
                <MovieCard key={idx} movie={r} />
            ))}
        </div>
    );
}
