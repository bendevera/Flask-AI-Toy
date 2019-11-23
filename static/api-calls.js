$('#sentiment-submit').click(() => {
    let query = $('#sentiment-query').val();
    if (query == '') {
        flash('Must input text to query sentiment classifier.');
    } else {
        let url = 'http://127.0.0.1:5000/api/sentiment?query=' + query;
        console.log('Querying url: ' + url);
        $.ajax({
            url: url,
            success: function(data) {
                console.log(data);
                let sentiment = data['prediction']
                let confidence = data['confidence']
                $('#neg-pos').html(sentiment);
                $('#sentiment-confidence').html(confidence);
            }
        })
    }
})

$('#diamond-submit').click(() => {
    let carat = $('#diamond-carat').val();
    let cut = $('#diamond-cut').val();
    console.log('Query is carat: ' + carat + ' cut: ' + cut);
    if (carat === null) {
        flash('Must input carat value to get prediction.');
    } else {
        let url = 'http://127.0.0.1:5000/api/diamond?carat=' + carat + '&cut=' + cut;
        console.log('Querying url: ' + url);
        $.ajax({
            url: url,
            success: function(data) {
                console.log(data);
                let price = data['prediction']
                $('#diamond-price').html(price);
            }
        })
    }
})