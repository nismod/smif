import React from 'react'
import { expect } from 'chai'
import { shallow } from 'enzyme'
import ProjectOverviewItem from '../../../../src/components/ConfigForm/ProjectOverview/ProjectOverviewItem.js'

import {sos_model_run, sos_models, narratives, sos_model} from '../../../helpers.js'
import {empty_object, empty_array} from '../../../helpers.js'

var wrapper, item

var itemname = "item_name"
var items = [
    {
        name: "item_1",
        description: "item_description_1"
    },
    {
        name: "item_2",
        description: "item_description_2"
    }
]
var itemlink = "/item/link/"

describe('<ProjectOverviewItem />', () => {

    it('renders itemname', () => {
        wrapper = shallow(<ProjectOverviewItem itemname={itemname} items={items} itemLink={itemlink} />)

        item = wrapper.find('[id="row_item_1"]').first()
        expect(item.html()).to.contain('value="' + itemname + '"')
    })

    it('renders items', () => {
        wrapper = shallow(<ProjectOverviewItem itemname={itemname} items={items} itemLink={itemlink} />)

        item = wrapper.find('[id="row_item_1"]')
        expect(item.html()).to.contain('<td id="item_1">' + items[0].name + '</td><td id="item_1">' + items[0].description + '</td>')

        item = wrapper.find('[id="row_item_2"]')
        expect(item.html()).to.contain('<td id="item_2">' + items[1].name + '</td><td id="item_2">' + items[1].description + '</td>')
    })

    it('warning no itemname', () => {
        wrapper = shallow(<ProjectOverviewItem itemname={""} items={items} itemLink={itemlink} />)

        item = wrapper.find('[id="project_overview_item_alert-danger"]')
        expect(item.html()).to.contain('There is no itemname configured')

        wrapper = shallow(<ProjectOverviewItem itemname={null} items={items} itemLink={itemlink} />)

        item = wrapper.find('[id="project_overview_item_alert-danger"]')
        expect(item.html()).to.contain('There is no itemname configured')
    })

    it('warning no items', () => {
        wrapper = shallow(<ProjectOverviewItem itemname={itemname} items={empty_array} itemLink={itemlink} />)

        item = wrapper.find('[id="project_overview_item_alert-info"]')
        expect(item.html()).to.contain('There are no items in this list')

        wrapper = shallow(<ProjectOverviewItem itemname={itemname} items={null} itemLink={itemlink} />)

        item = wrapper.find('[id="project_overview_item_alert-info"]')
        expect(item.html()).to.contain('There are no items in this list')
    })

    it('warning no itemLink', () => {
        wrapper = shallow(<ProjectOverviewItem itemname={itemname} items={items} itemLink={""} />)

        item = wrapper.find('[id="project_overview_item_alert-danger"]')
        expect(item.html()).to.contain('There is no itemLink configured')

        wrapper = shallow(<ProjectOverviewItem itemname={itemname} items={items} itemLink={null} />)

        item = wrapper.find('[id="project_overview_item_alert-danger"]')
        expect(item.html()).to.contain('There is no itemLink configured')
    })
})
