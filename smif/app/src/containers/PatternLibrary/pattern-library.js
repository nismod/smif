import React from 'react';

import FaTrash from 'react-icons/lib/fa/trash';

import './pattern-library.css';

export default class PatternLibrary extends React.Component {
    // 'Hello World' render to style guide page
    // - should be able to visit at http://localhost:8080/style-guide.html
    //   when running the dev server (npm start)
    // - TODO move this component to a suitable directory (containers)
    // - TODO add light-touch style-guide-specific styles
    render() {
        return (
            <article>
                <header className="guide-header" id="attr-pattern-lib">
                    <h1>smif &ndash; Pattern library</h1>
                </header>
                <div className="guide-content">
                    <h2>Typography</h2>
                    <div className="sample-container">
                        <h1>Heading Level 1</h1>
                    </div>
                    <div className="sample-container">
                        <h2>Heading Level 2</h2>
                    </div>
                    <div className="sample-container">
                        <h3>Heading Level 3</h3>
                    </div>
                    <div className="sample-container">
                        <p>
                            ITRC is a consortium of seven leading UK universities,
                            investigating ways to improve the performance of infrastructure
                            systems in the UK and around the world. Our research is helping
                            businesses and policymakers to explore the risk of infrastructure
                            failure and the long term benefits of investments and policies
                            to improve infrastructure systems.
                        </p>
                    </div>
                    <div className="sample-container">
                        <i>Italic</i>
                    </div>
                    <div className="sample-container">
                        <b>Bold</b>
                    </div>

                    <h2>Inline text elements</h2>
                    <h3>Abbreviation</h3>
                    <div className="sample-container">
                        <abbr title="abbreviation">abbr.</abbr>
                    </div>
                    <h3>Code</h3>
                    <div className="sample-container">
                        <code>var i = 0;</code>
                    </div>
                    <h3>Details</h3>
                    <div className="sample-container">
                        <details>
                            <summary>Some details</summary>
                            <p>More info about the details.</p>
                        </details>
                    </div>
                    <h3>External link</h3>
                    <div className="sample-container">
                        <a href="https://www.itrc.org.uk/">Link to the ITRC homepage</a>
                    </div>
                    <h3>Same-page link</h3>
                    <div className="sample-container">
                        <a href="#attr-pattern-lib">Link to the header of this page</a>
                    </div>
                    <h3>Fieldset</h3>
                    <fieldset>
                        <legend>Fieldset</legend>
                    </fieldset>

                    <h2>Input elements</h2>
                    <h3>Button</h3>
                    <div className="sample-container">
                        <input type="button" value="Click Me" />
                    </div>
                    <h3>Checkbox</h3>
                    <div className="sample-container">
                        <label>
                            <input type="checkbox" />
                            Subscribe to newsletter?
                        </label>
                    </div>
                    <h3>Date</h3>
                    <div className="sample-container">
                        <input type="date" value="2017-06-01"/>
                    </div>
                    <h3>Datetime</h3>
                    <div className="sample-container">
                        <input type="datetime-local" value="2017-06-01T08:30" />
                    </div>
                    <h3>File</h3>
                    <div className="sample-container">
                        <input type="file" />
                    </div>
                    <h3>Number</h3>
                    <div className="sample-container">
                        <input type="number" />
                    </div>
                    <h3>Radio Button</h3>
                    <div className="sample-container">
                        <label>
                            <input type="radio" />
                            Subscribe to newsletter?
                        </label>
                    </div>
                    <h3>Range</h3>
                    <div className="sample-container">
                        <input type="range" />
                    </div>
                    <h3>Text Field</h3>
                    <div className="sample-container">
                        <input type="text" />
                    </div>
                    <h3>Text Area</h3>
                    <div className="sample-container">
                        <textarea name="textarea" rows="10" cols="50">Write something here</textarea>
                    </div>
                    <h3>Droplist</h3>
                    <div className="sample-container">
                        <select>
                            <option value="" disabled="disabled" selected="selected">Please select a name</option>
                            <option value="1">One</option>
                            <option value="2">Two</option>
                        </select>
                    </div>
                    <h3>Datalist</h3>
                    <div className="sample-container">
                        <label>Choose a model:
                            <input type="text" list="models" name="myModels" />
                        </label>
                        <datalist id="models">
                            <option value="Water">Water</option>
                            <option value="Energy-Demand">Energy Demand</option>
                            <option value="Energy-Supply">Energy Supply</option>
                            <option value="Solid-Waste">Solid Waste</option>
                        </datalist>
                    </div>
                    <h3>Select Menu</h3>
                    <div className="sample-container">
                        <select size="3">
                            <option value="Water">Water</option>
                            <option value="Energy-Demand">Energy Demand</option>
                            <option value="Energy-Supply">Energy Supply</option>
                            <option value="Solid-Waste">Solid Waste</option>
                        </select>
                    </div>

                    <h2>Display elements</h2>
                    <h3>Table</h3>
                    <div className="sample-container">
                        <table>
                            <tr>
                                <th>Firstname</th>
                                <th>Lastname</th>
                            </tr>
                            <tr>
                                <td>Roald</td>
                                <td>Lemmen</td>
                            </tr>
                            <tr>
                                <td>Will</td>
                                <td>Usher</td>
                            </tr>
                            <tr>
                                <td>Tom</td>
                                <td>Russell</td>
                            </tr>
                        </table>
                    </div>

                    <h2>Icons</h2>
                    <h3>Delete</h3>
                    <div className="sample-container">
                        <FaTrash />
                    </div>
                </div>

            </article>
        );
    }
}
