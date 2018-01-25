import React from 'react'
import { expect } from 'chai'
import { shallow } from 'enzyme'
import ScenarioConfigForm from '../../../src/components/ConfigForm/ScenarioConfigForm.js'

import {scenario, scenario_sets} from '../../helpers.js'
import {empty_object, empty_array} from '../../helpers.js'

describe('<ScenarioConfigForm />', () => {

    const correctRender = shallow(<ScenarioConfigForm scenario={scenario} scenarioSets={scenario_sets} />)
    const dataMissingRender = shallow(<ScenarioConfigForm scenario={empty_object} scenarioSets={empty_array} />)

    it('renders scenario.name', () => {
        const scenario_name = correctRender.find('[id="scenario_name"]')
        expect(scenario_name.html()).to.contain(scenario.name)
    })

    it('renders scenario.name when data missing', () => {
        const scenario_name = dataMissingRender.find('[id="scenario_name"]')
        expect(scenario_name.html()).to.contain(`id="scenario_name"`)
    })

    it('renders scenario.description', () => {
        const scenario_description = correctRender.find('[id="scenario_description"]')
        expect(scenario_description.html()).to.contain(scenario.description)
    })

    it('renders scenario.description when data missing', () => {
        const scenario_description = dataMissingRender.find('[id="scenario_description"]')
        expect(scenario_description.html()).to.contain(`id="scenario_description"`)
    })
})
