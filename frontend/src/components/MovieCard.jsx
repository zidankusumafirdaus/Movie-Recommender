import React from 'react';
import { genreIdToName } from '../utils/genres';

export default function MovieCard({ movie }) {
    const rawGenres = movie.movie_genres || '';
    const genres = rawGenres.replace(/^\[|]$/g, '').trim().split(/\s+/).filter(Boolean).map(g => genreIdToName(g));

    return (
        <div className="flex gap-4 bg-gradient-to-br from-primary-500/30 to-primary-900/20 rounded-xl p-4 shadow-lg items-start">
            <div className="w-20 h-28 bg-primary-900/80 rounded-md flex-shrink-0 flex items-center justify-center text-primary-300">Poster</div>

            <div className="flex-1">
                <div className="flex items-start justify-between">
                    <div>
                        <h3 className="font-semibold text-lg text-white leading-tight">{movie.movie_title}</h3>
                        {movie.movie_id && (
                            <p className="text-xs text-primary-300 mt-1">ID: <span className="font-mono text-primary-300">{movie.movie_id}</span></p>
                        )}
                    </div>

                    {movie.predicted_rating !== undefined && movie.predicted_rating !== null && (
                        <div className="text-right">
                            <div className="text-2xl font-bold text-primary-300">{Number(movie.predicted_rating).toFixed(1)}</div>
                            <div className="text-xs text-gray-300">predicted</div>
                        </div>
                    )}
                </div>

                {genres.length > 0 && (
                    <div className="flex flex-wrap gap-2 mt-3">
                        {genres.map((g, i) => (
                            <span key={i} className="text-xs bg-primary-500/20 text-primary-300 px-2 py-1 rounded-full">{g}</span>
                        ))}
                    </div>
                )}
            </div>
        </div>
    );
}
