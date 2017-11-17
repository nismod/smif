import React from 'react';

const NarrativeSelector = ({narrativeSet, narratives}) => (
    <fieldset>
        <legend>{narrativeSet}</legend>
        {
            narratives.map((narrative) => (
                <div key={narrative.name}>
                    <input type="checkbox" key={narrative.name} value={narrative.name} defaultChecked={narrative.active}></input>
                    <label>{narrative.name}</label>
                </div>
            ))
        }
    </fieldset>
);

export default NarrativeSelector;