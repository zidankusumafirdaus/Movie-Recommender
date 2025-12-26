import React, { useState, useEffect } from 'react';
import '../styles/index.css';
import { getMovies } from '../api/api';
import MoviesList from '../components/MoviesList';
import RatingsSidebar from '../components/HistoryRatings';
import Navbar from '../components/Navbar';

export default function Movies() {
    const [movies, setMovies] = useState([]);
    const [showCount, setShowCount] = useState(50);
    const [, setLoadingMovies] = useState(false);

    const fetchMovies = async () => {
        setLoadingMovies(true);
        try {
            const res = await getMovies();
            setMovies(res.data.movies || []);
            setShowCount(50);
        } catch (err) {
            console.error(err);
            setMovies([]);
        } finally {
            setLoadingMovies(false);
        }
    };

    useEffect(() => {
        fetchMovies();
    }, []);



    return (
        <div className="min-h-screen bg-primary-900 text-white p-6">
            <Navbar />
            <div className="max-w-1xl mx-auto mt-6">
                <section className="grid grid-cols-1 lg:grid-cols-3 gap-4">
                    <div className="lg:col-span-2">
                        <div className="flex items-center justify-between mb-4">
                            <div>
                                <h2 className="text-4xl font-extrabold">Browse Movies</h2>
                                <p className="text-gray-200 mt-2">Explore the catalog and submit ratings to improve recommendations.</p>
                            </div>
                        </div>

                        <div className="space-y-4">
                            {movies.length === 0 ? (
                                <MoviesList movies={[]} />
                            ) : (
                                <>
                                    <MoviesList movies={movies.slice(0, showCount)} />
                                    <div className="flex justify-center mt-2">
                                        {movies.length > showCount ? (
                                            <button onClick={() => setShowCount(prev => Math.min(prev + 50, movies.length))} className="bg-primary-500 hover:bg-primary-700 text-white py-2 px-4 rounded-md shadow">Load more</button>
                                        ) : (
                                            movies.length > 50 && (
                                                <button onClick={() => setShowCount(50)} className="bg-primary-700 hover:bg-primary-600 text-white py-2 px-4 rounded-md shadow">Show less</button>
                                            )
                                        )}
                                    </div>
                                </>
                            )}
                        </div>
                    </div>

                    <RatingsSidebar initialUserId="100" />
                </section>
            </div>
        </div>
    );
}
