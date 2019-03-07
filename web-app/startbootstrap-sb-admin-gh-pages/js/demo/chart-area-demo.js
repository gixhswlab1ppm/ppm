// Firebase stuff

// // Auth user
// var userId = firebase.auth().currentUser.uid;

// // Get a reference to the storage service, which is used to create references in your storage bucket
// var storage = firebase.storage();

// // Create a storage reference from our storage service
// var storageRef = storage.ref();

var data;
var labels;
var impact;
var accel_x;
var scaled_impact;

// from https://qiita.com/coffee_and_code/items/72f00581c032693c6e33
firebase.auth().signInWithEmailAndPassword('opsec@google.com', 'opsec1').catch((error) => {
    console.log('code:' + error.code + 'message' + error.message);
});
firebase.auth().onAuthStateChanged(function (user) {
    if (user) {
        console.log('login success');
    } else { }
});

var raw;

var storage = firebase.storage();
storageRef = storage.ref('');
// console.log("1", storageRef);
// inits a variable to linearly search down from
var i = 55;
// tries the next file
// TODO: replace this approach with latest_file() instead of i.toString()
try_next_file();
function try_next_file() {
  console.log('accessing file: '+i.toString()+'.json');
  storageRef.child(i.toString()+'.json').getDownloadURL().then(function (url) {
      var xhr = new XMLHttpRequest();
      xhr.responseType = 'json';
      // asynch stuff, this is basically our main
      xhr.onload = function (event) {
          // parse data once you've got it
          parse_data(xhr.response);

          // Call line_chart on data with scaling, see above
          line_chart("impactChart", "Impact Strength", impact, 0, 
            Math.ceil(impact.reduce(function(a, b) { return Math.max(a, b); }) / 8) * 8);
          // WIP
          bar_chart("impactBarChart", "Impact Strength", impact_bar, 0, 
            Math.ceil(impact_bar["datasets"][0]["data"].reduce(function(a, b) { return Math.max(a+5, b+5); })));
          // Call line_chart on acceleration data
          line_chart("accelerationChart", "Acceleration Speed", accel_x, 
            accel_x.reduce(function(a, b) { return Math.min(a, b); }), accel_x.reduce(function(a, b) { return Math.max(a, b); }));
      };
      xhr.open('GET', url);
      xhr.send();
  }).catch(function (error) {
      // Report errors to log
      console.log("error caught", error);
      // part of try_next_file code
      i--;
      try_next_file()
      // end of try_next_file code
      // Handle any errors
  });
}

// End Firebase stuff

// Docs at https://www.chartjs.org/docs/latest/

// Set new default font family and font color to mimic Bootstrap's default styling
Chart.defaults.global.defaultFontFamily = '-apple-system,system-ui,BlinkMacSystemFont,"Segoe UI",Roboto,"Helvetica Neue",Arial,sans-serif';
Chart.defaults.global.defaultFontColor = '#292b2c';

/* Function latest_file figures out the most recent file
   in the Firebase Storage which is active */
function latest_file(storage) {
  // see https://cloud.google.com/nodejs/docs/reference/storage/1.7.x/Bucket
  // and https://cloud.google.com/nodejs/docs/reference/storage/1.7.x/Bucket.html#getFiles
  // including require in HTML index file broke chart.js somehow
  // require(['@google-cloud/storage'], function (gcs) {
  //     //gcs is now loaded.
  // });
  // console.log("what files exist");
  // var query = {
  //   directory: '/'
  // };
  // gcs.bucket(storage.bucket).getFiles(query, function(err, files, nextQuery, apiResponse) {
  //    console.log("here")
  //    console.log(files);
  // });
}

/* Function parse_data takes in the json file from the server
   and parses the files into global variables declared as such:
    var data = {};
    var labels = [];
    var impact = [];
    var accel_x = [];
    var scaled_impact; */
function parse_data(raw) {
  data = {};
  labels = [];
  impact = [];
  accel_x = [];
  for (i = 0; i < raw.length; i++) {
    data[i] = {
      time: raw[i][0]/1000,
      a_x: raw[i][1],
      a_y: raw[i][2],
      a_z: raw[i][3],
      g_x: raw[i][4],
      g_y: raw[i][5],
      g_z: raw[i][6],
      p_1: raw[i][7],
      p_2: raw[i][8],
      p_3: raw[i][9],
    }
  }
  for (var key in data) {
    // console.log(data[key])
    // console.log(data[key]["time"])
    
    // Create a time-series array for x axis for time series
    labels.push(data[key]["time"])
    
    /*  Calculate how hard the ball hit and 
        add it to an array for y axis */
    scaled_impact = Math.log(data[key]["p_1"] + data[key]["p_1"] + data[key]["p_1"] / 3)
    impact.push(Math.round((scaled_impact) * 100) / 100)
    // average out bad data
    if (impact[impact.length - 2] == Number.NEGATIVE_INFINITY) {
      impact[impact.length - 2] = (impact[impact.length - 1] + impact[impact.length - 3]) / 2
        // consecutive occurance of bad data gets set to 0
      if (impact[impact.length - 2] == Number.NEGATIVE_INFINITY) {
        impact[impact.length - 2] = 0
      }
    }

    // catch zeroes and negative infinities from final list
    for (var i = 0; i < impact.length; i++) {
      if (impact[i] == Number.NEGATIVE_INFINITY) {
        impact[i] = 0;
      }
      else if (isNaN(impact[i])) {
        impact[i] = 0;
      }
    }

    // Add basic data to array for printing
    accel_x.push(Math.round((data[key]["a_x"]) * 1000) / 10)
  }

  // defines dict for strong/weak thresholds for bar chart
  impact_bar = {
    labels: ["Strong", "Weak"],
    datasets: [{
      label: "Data",
      backgroundColor: "rgba(2,117,216,0.2)",
      borderColor: "rgba(2,117,216,1)",
      borderWidth: 1,
      data: [0, 0]}]
  }

  // thresholding code to determine how many strong hits vs how many weak hits occur
  for (i = 0; i < impact.length; i++) {
    if (impact[i] > 4) {impact_bar["datasets"][0]["data"][0] += 1;}
    else if (impact[i] > 2.8) {impact_bar["datasets"][0]["data"][1] += 1;}
  }
}

// console.log(labels)
// console.log(data)
// console.log(impact)
// console.log(accel_x)

/*  line_chart requires ID of a canvas in DOM,
    label for the values,
    data where len(data) == len(timestamps),
    and minimum and maximum scales for y axis */
function line_chart(chart_name, item_label, data, minScale, maxScale) {
  var ctx = document.getElementById(chart_name);
  var myLineChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels: labels,
      datasets: [{
        fill: false,
        label: item_label,
        lineTension: 0.3,
        backgroundColor: "rgba(2,117,216,0.2)",
        borderColor: "rgba(2,117,216,1)",
        pointRadius: 3,
        pointBackgroundColor: "rgba(2,117,216,1)",
        pointBorderColor: "rgba(255,255,255,0.8)",
        pointHoverRadius: 3,
        pointHoverBackgroundColor: "rgba(2,117,216,1)",
        pointHitRadius: 50,
        pointBorderWidth: 2,
        data: data,
        spanGaps: true,
      }],
    },
    options: {
      scales: {
        xAxes: [{
          time: {
            unit: 'seconds'
          },
          gridLines: {
            display: false
          },
          ticks: {
            maxTicksLimit: 7,
            callback: function(value, index, values) {
              return (Math.round((value-labels[0]) * 100) / 100).toString() + 's';
            },
          }
        }],
        yAxes: [{
          ticks: {
            min: minScale,
            max: maxScale,
            maxTicksLimit: 5,
          },
          gridLines: {
            color: "rgba(0, 0, 0, .125)",
          }
        }],
      },
      legend: {
        display: false
      }
    }
  });
}

// wip
/*  bar_chart requires ID of a canvas in DOM,
    label for the values,
    data where len(data) == len(timestamps),
    and minimum and maximum scales for y axis */
function bar_chart(chart_name, item_label, data, minScale, maxScale) {
  var ctx = document.getElementById(chart_name);
  var myLineChart = new Chart(ctx, {
    type: 'bar',
    data: data,
    options: {
      scales: {
        xAxes: [{
          gridLines: {
            display: true,
          },
          ticks: {
            maxTicksLimit: 2,
          }
        }],
        yAxes: [{
          ticks: {
            min: minScale,
            max: maxScale,
            maxTicksLimit: 5,
          },
          gridLines: {
            color: "rgba(0, 0, 0, .125)",
          }
        }],
      },
      legend: {
        display: false,
      }
    }
  });
}