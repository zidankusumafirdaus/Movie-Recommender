import React from 'react';
import MovieCard from './MovieCard';

export default function MoviesList({ movies }) {
    if (!movies || movies.length === 0) {
        return <p className="text-center text-primary-300">No movies available.</p>;
    }

    return (
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-2 gap-4">
            {movies.map((m) => (
                <MovieCard key={m.movie_id} movie={m} />
            ))}
        </div>
    );
}
