import {createBottomTabNavigator} from "react-navigation-tabs"
import {FontAwesome, AntDesign} from '@expo/vector-icons';
import {GetStarted, AuthScreen, AddItem, Items} from "../screens"
import {createStackNavigator} from "react-navigation-stack"
import React from "react";

export const SplashScreen = createStackNavigator(
    {
        GetStarted,
        AuthScreen
    },
    {
        defaultNavigationOptions: {
            header: null
        }
    }
);


export const ItemScreens = createBottomTabNavigator({
    Items,
    AddItem,
}, {
    defaultNavigationOptions: ({navigation}) => ({
        tabBarIcon: ({focused, horizontal, tintColor}) => {
            const {routeName} = navigation.state;
            let IconComponent = FontAwesome;
            let iconName;
            if (routeName === 'Items') {
                iconName = `product-hunt`;
                return <IconComponent name={iconName} size={20} color={tintColor}/>;
            } else if (routeName === 'AddItem') {
                iconName = `addfile`;
                return <AntDesign name={iconName} size={20} color={tintColor}/>
            }

        },
    }),
    tabBarOptions: {
        activeTintColor: '#283593',
        labelStyle: {
            fontSize: 10,
        },
    }

});
