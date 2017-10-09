import React from 'react'

const SosModelConfig = () => (
    <div>
        <form>
            <h2>Create System-of-systems Model</h2>

            <p>Name</p>
            <input name="sos_model_name" type="text"/><br />

            <p>Description</p>
            <textarea name="sos_model_description" rows="10" cols="30"/><br />

            <table>
            </table>

            <p>Scenario Sets</p>

            <p>Sector Models</p>

            <p>Dependencies</p>

            <button type="submit">
                Submit
            </button>
    </form>
  </div>
)

export default SosModelConfig
