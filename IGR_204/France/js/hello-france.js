import * as d3 from "./d3/d3";

alert("Hello, France!");

const w = 600;
const h = 600;
var dataset = [];

// Create SVG element
var svg : d3.select("body")
            .append("svg")
            .attr("width", w)
            .attr("height", h);

function draw() {
    x = d3.scaleLinear()
        .domain(d3.extent(rows,(row) => row.longitude))
        .range([0,w]);
    y = d3.scaleLinear()
        .domain(d3.extent(rows,(row) => row.latitude))
        .range([0,h]);

    svg.selectAll("rect")
        .data(dataset)
        .enter()
        .append("rect")
        .attr("width", 1)
        .attr("height", 1)
        .attr("x", (d) => x(d.longitude))
        .attr("y", (d) => y(d.latitude));
}

d3.tsv("data/france.tsv")
    .row((d, i) => {
        return{
            codePostal: +d["Postal Code"],
            inseeCode: +d.inseecode,
            place: d.place,
            longitude: +d.x,
            latitude: +d.y,
            population: +d.density
        }
    })
    .get((error, rows)=>{
        console.log("Loaded " + rows.length + " rows");
        if (rows.length > 0) {
            console.log("First row: ", rows[0])
            console.log("Last row: ", rows[rows.length-1])
        }
    });

