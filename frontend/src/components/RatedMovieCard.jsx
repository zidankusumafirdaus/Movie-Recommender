import React from 'react';
import { genreIdToName } from '../utils/genres';

export default function RatedMovieCard({ movie }) {
    const rawGenres = movie.movie_genres || '';
    const genres = rawGenres.replace(/^\[|\]$/g, '').trim().split(/\s+/).filter(Boolean).map(g => genreIdToName(g));
    const date = movie.timestamp ? new Date(movie.timestamp * 1000).toLocaleString() : null;

    return (
        <div className="flex gap-3 items-start bg-primary-900/25 rounded-lg p-3 shadow-sm">
            <div className="w-16 h-24 bg-primary-700/40 rounded-md flex items-center justify-center text-primary-300">Poster</div>
            <div className="flex-1">
                <div className="flex items-center justify-between">
                    <div>
                        <h3 className="font-semibold text-md text-white">{movie.movie_title}</h3>
                        {movie.movie_id && (
                            <p className="text-xs text-primary-300 mt-1">ID: <span className="font-mono">{movie.movie_id}</span></p>
                        )}
                    </div>

                    <div className="text-right">
                        <div className="text-lg font-bold text-primary-300">{movie.user_rating}</div>
                        <div className="text-xs text-gray-300">your rating</div>
                    </div>
                </div>

                {genres.length > 0 && (
                    <div className="flex flex-wrap gap-2 mt-2">
                        {genres.map((g, i) => (
                            <span key={i} className="text-xs bg-primary-500/20 text-primary-300 px-2 py-1 rounded-full">{g}</span>
                        ))}
                    </div>
                )}

                {date && <p className="text-xs text-gray-300 mt-2">Rated at: {date}</p>}
            </div>
        </div>
    );
}
