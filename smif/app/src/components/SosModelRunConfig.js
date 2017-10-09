import React from 'react';

const SosModelRunConfig = () => (
    <div>
        <form>
            <h2>Create System-of-systems Model-Run Configuration</h2>

            <p>Name</p>
            <input name="project_name" type="text"/>

            <p>Description</p>
            <textarea name="sos_model_description" rows="10" cols="30"/>

            <p>System-of-systems Model</p>

            <p>Decisions Module</p>

            <p>Scenarios</p>

            <p>Narratives</p>

            <button type="submit">
                Submit
            </button>
        </form>
    </div>
);

export default SosModelRunConfig;
