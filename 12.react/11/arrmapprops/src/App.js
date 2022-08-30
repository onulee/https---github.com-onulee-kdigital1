import logo from './logo.svg';
import './App.css';
import React from 'react';
import Movie from './components/Movie';

function App() {
  const movies = [
    { title: '아이언맨1', year: 2008},
    { title: '아이언맨2', year: 2010},
    { title: '아이언맨3', year: 2012},
    { title: '아이언맨4', year: 2014},
  ];
  const renderMovies = movies.map(movie => {
    return (
      <Movie movie={movie} key={movie.title}/>
    );
  });
  return (
    <div className="App">
      <h1>Movie list</h1>
      {renderMovies}
    </div>
  );
}

export default App;