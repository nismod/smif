import React from 'react';

export default class App extends React.Component {
    render() {
        return (
            <div className="content-wrapper">
                <h1>{this.props.heading}</h1>
            </div>);
    }
}
