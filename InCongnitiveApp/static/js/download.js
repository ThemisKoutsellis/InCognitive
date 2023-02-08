

function dict_to_xlsx(data){

    //Find max length
    const nodes_order_max = Math.max(data['nodes_order'].length, data['nodes_discription'].length, data['initial_values'].length,data['auto_weights'].length,data['auto_lags'].length )

    //Create AoA from dict data for each sheet
    //If an array is shorter than max length and have no data puts empty instead
    var nodes_order_aoa = [];
    for (var i = 0; i < nodes_order_max; i++ ){
        nodes_order_aoa[i] = [data['nodes_order'][i]!=null?data['nodes_order'][i]:"", 
                            data['nodes_discription'][i]!=null?data['nodes_discription'][i]:"",
                            data['initial_values'][i]!=null?data['initial_values'][i]:"",
                            data['auto_weights'][i]!=null?data['auto_weights'][i]:"",
                            data['auto_lags'][i]!=null?data['auto_lags'][i]:"" 
                        ]
    }


    const io_nodes_max = Math.max(data['nodes_order'].length, data['nodes_discription'].length);

    var io_nodes_aoa = [];
    for (var i = 0; i < io_nodes_max; i++ ){
        io_nodes_aoa[i] = [data['input_nodes'][i]!=null?data['input_nodes'][i]:"", 
                            data['output_nodes'][i]!=null?data['output_nodes'][i]:""
                        ]
    }


    const fcm_topology_max = Math.max(data['source_nodes'].length, data['target_nodes'].length, data['weights'].length, data['lags'].length);

    var fcm_topology_aoa = [];
    for (var i = 0; i < fcm_topology_max; i++ ){
        fcm_topology_aoa[i] = [data['source_nodes'][i]!=null?data['source_nodes'][i]:"", 
                            data['target_nodes'][i]!=null?data['target_nodes'][i]:"",
                            data['weights'][i]!=null?data['weights'][i]:"",
                            data['lags'][i]!=null?data['lags'][i]:""
                        ]
    }

    //Create worksheets from AoA and puts Headers
    const nodes_order_worksheet = XLSX.utils.aoa_to_sheet(nodes_order_aoa);
    XLSX.utils.sheet_add_aoa(nodes_order_worksheet, [["nodes order", "node description", "initial value", "auto weights","auto lags"]], { origin: "A1" });
    const io_nodes_worksheet = XLSX.utils.aoa_to_sheet(io_nodes_aoa);
    XLSX.utils.sheet_add_aoa(io_nodes_worksheet, [["input nodes", "output nodes"]], { origin: "A1" });
    const fcm_topology_worksheet = XLSX.utils.aoa_to_sheet(fcm_topology_aoa);
    XLSX.utils.sheet_add_aoa(fcm_topology_worksheet, [["source node", "target node", "weight", "lag"]], { origin: "A1" });

    //Create workbook and append sheets
    var wb = XLSX.utils.book_new();
    XLSX.utils.book_append_sheet(wb, nodes_order_worksheet, "nodes-order");
    XLSX.utils.book_append_sheet(wb, io_nodes_worksheet, "input-output_nodes");
    XLSX.utils.book_append_sheet(wb, fcm_topology_worksheet, "fcm-topology");


    return wb;
}


const data = JSON.parse(cb_obj.text);

//write workbook to unit8 buffer and then create xlsx blob
const u8 = XLSX.write(dict_to_xlsx(data), { type: "buffer", bookType: "xlsx" });
const blob = new Blob([u8], { type: "application/vnd.ms-excel" });

const filename = 'output.xlsx';

//Triggers browser download

//addresses IE
if (navigator.msSaveBlob) {
    navigator.msSaveBlob(blob, filename);
}

else {
    const link = document.createElement('a')
    link.href = URL.createObjectURL(blob);
    link.download = filename
    link.target = "_blank";
    link.style.visibility = 'hidden';
    link.dispatchEvent(new MouseEvent('click'))
}
