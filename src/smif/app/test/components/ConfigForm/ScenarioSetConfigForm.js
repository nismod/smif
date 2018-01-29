import React from 'react'
import { expect } from 'chai'
import { shallow } from 'enzyme'
import ScenarioSetConfigForm from '../../../src/components/ConfigForm/ScenarioSetConfigForm.js'

import {scenario_set} from '../../helpers.js'
import {empty_object, empty_array} from '../../helpers.js'

describe('<ScenarioSetConfigForm />', () => {

    const correctRender = shallow(<ScenarioSetConfigForm scenarioSet={scenario_set} />)
    const dataMissingRender = shallow(<ScenarioSetConfigForm scenarioSet={empty_object} />)

    it('renders scenario_set.name', () => {
        const scenario_set_name = correctRender.find('[id="scenario_set_name"]')
        expect(scenario_set_name.html()).to.contain(scenario_set.name)
    })

    it('renders scenario_set.name when data missing', () => {
        const scenario_set_name = dataMissingRender.find('[id="scenario_set_name"]')
        expect(scenario_set_name.html()).to.contain(`id="scenario_set_name"`)
    })

    it('renders scenario_set.description', () => {
        const scenario_set_description = correctRender.find('[id="scenario_set_description"]')
        expect(scenario_set_description.html()).to.contain(scenario_set.description)
    })

    it('renders scenario_set.description when data missing', () => {
        const scenario_set_description = dataMissingRender.find('[id="scenario_set_description"]')
        expect(scenario_set_description.html()).to.contain(`id="scenario_set_description"`)
    })
})
