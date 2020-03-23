import React from "react";
import Constants from 'expo-constants';
import {View, Text, ScrollView, FlatList, ActivityIndicator, TouchableOpacity, Modal} from "react-native"
import axios from "axios";
import {LinearGradient} from "expo-linear-gradient";

function ItemCard({product_name, mfg_date, exp_date, mrp, prediction}) {
    return (
        <>

            <View style={{marginBottom: 10}}>
                <Text style={{textAlign: "center", color: "#283593", fontWeight: "bold"}}>
                    {product_name}
                </Text>
            </View>
            <View
                style={{flexDirection: "row", justifyContent: "space-between", marginBottom: 10, flexWrap: 'wrap'}}>
                <Text>
                    MFG Date: {mfg_date}
                </Text>
                <Text>
                    EXP Date: {exp_date}
                </Text>
            </View>
            <View
                style={{flexWrap: "wrap", flexDirection: "row", justifyContent: "space-between", marginBottom: 10}}>
                <Text style={{fontWeight: "bold"}}>
                    MRP: {mrp}
                </Text>
                <Text style={{color: "red"}}>
                    Dynamic Pricing: {prediction}
                </Text>
            </View>
        </>
    )
}

function Item({product_name, mfg_date, exp_date, mrp, prediction}) {
    const [modal, setModal] = React.useState(false);
    return <>
        <TouchableOpacity
            // onPress={() => setModal(true)}
            style={{
                marginBottom: 20,
                flex: 1,
                padding: 10,
                borderRadius: 10,
                elevation: 2,
                backgroundColor: "white"
            }}>
            <ItemCard {...{product_name, mfg_date, exp_date, mrp, prediction}} />
        </ TouchableOpacity>
        <Modal
            animationType="slide"
            transparent={false}
            visible={modal}
            onRequestClose={() => setModal(false)}>
            <View style={{
                marginBottom: 20,
                flex: 1,
                padding: 10,
                borderRadius: 10,
                elevation: 2,
                backgroundColor: "white"
            }}>
                <ItemCard {...{product_name, mfg_date, exp_date, mrp, prediction}} />
            </View>
        </Modal>
    </>
}


export default function Items(props) {
    const [allItems, setItems] = React.useState([]);

    const axiosCall = () => {
        return axios.get("https://foodkart-abf29.firebaseio.com/item.json")
            .then(({data}) => {
                let requiredAr = [];
                for (let i in data) {
                    for (let x of data[i]) {
                        const {mfg_date, exp_date, mrp} = x;
                        axios.post("https://foodkart-undefined.azurewebsites.net/predict", [{
                            mfg_date,
                            exp_date,
                            mrp
                        }])
                            .then(({data: {prediction}}) => {
                                requiredAr = [
                                    ...requiredAr,
                                    {
                                        ...x,
                                        prediction
                                    }
                                ]
                                setItems(requiredAr)
                            })
                            .catch(res => console.log("here 2", {res}, {mfg_date, exp_date, mrp}))
                    }

                }
            })
            .catch(res => console.log("here", {res}))
    }


    React.useEffect(() => {
        const reloadBool = props.navigation.getParam("reload");
        if (reloadBool) {
            axiosCall().then(res => res)
                .catch(er => er)
        }
    }, [props]);

    React.useEffect(() => {
        axiosCall().then(res => res)
            .catch(er => er)
    }, []);


    return (

        <LinearGradient style={{
            flex: 1, paddingTop: Constants.statusHeight,
            paddingHorizontal: 10,
        }} colors={["#667eea", "#764ba2"]}>
            <ScrollView
                contentContainerStyle={{

                    flexGrow: 1
                }}
                style={{flex: 1}}>
                {
                    allItems.length
                        ? <FlatList data={allItems} keyExtractor={(item, index) => index.toString()}
                                    renderItem={({item}) => <Item {...item} />}
                        />
                        : <View style={{flex: 1, justifyContent: "center", alignItems: "center"}}>
                            <ActivityIndicator size="large" color="#0000ff"/>
                        </View>
                }
            </ScrollView>
        </LinearGradient>
    )
}
