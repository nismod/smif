import React from 'react';

const SosModelConfig = () => (
    <div>
        <form>
            <h2>Create Sector Model Configuration</h2>

            <p>Name</p>
            <input name="sector_model_name" type="text"/>

            <p>Description</p>
            <textarea name="sector_model_description" rows="10" cols="30"/>

            <p>Classname</p>

            <p>Path</p>

            <p>Inputs</p>

            <p>Outputs</p>

            <p>Parameters</p>

            <p>Interventions</p>

            <p>Initial Conditions</p>

            <button type="submit">
                Submit
            </button>
        </form>
    </div>
);

export default SosModelConfig;
