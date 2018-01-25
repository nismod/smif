import React from 'react'
import { expect } from 'chai'
import { shallow } from 'enzyme'
import NarrativeConfigForm from '../../../src/components/ConfigForm/NarrativeConfigForm.js'

import {narrative, narrative_sets} from '../../helpers.js'
import {empty_object, empty_array} from '../../helpers.js'

describe('<NarrativeConfigForm />', () => {

    const correctRender = shallow(<NarrativeConfigForm narrative={narrative} narrativeSets={narrative_sets} />)
    const dataMissingRender = shallow(<NarrativeConfigForm narrative={empty_object} narrativeSets={empty_array} />)

    it('renders narrative.name', () => {
        const narrative_name = correctRender.find('[id="narrative_name"]')
        expect(narrative_name.html()).to.contain(narrative.name)
    })

    it('renders narrative.name when data missing', () => {
        const narrative_name = dataMissingRender.find('[id="narrative_name"]')
        expect(narrative_name.html()).to.contain(`id="narrative_name"`)
    })

    it('renders narrative.description', () => {
        const narrative_description = correctRender.find('[id="narrative_description"]')
        expect(narrative_description.html()).to.contain(narrative.description)
    })

    it('renders narrative.description when data missing', () => {
        const narrative_description = dataMissingRender.find('[id="narrative_description"]')
        expect(narrative_description.html()).to.contain(`id="narrative_description"`)
    })
})
