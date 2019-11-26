function convertToSlug(Text)
{
    return Text
        .toLowerCase()
        .replace(/ /g,'-')
        .replace(/[^\w-]+/g,'')
        ;
}

$('#sentiment-submit').click(() => {
    let query = $('#sentiment-query').val();
    if (query == '') {
        alert('Must input text to query sentiment classifier.');
    } else {
        query = convertToSlug(query)
        let url = 'http://127.0.0.1:5000/api/sentiment?query=' + query;
        console.log('Querying url: ' + url);
        $.ajax({
            url: url,
            success: function(data) {
                if (data['success']) {
                    console.log(data)
                    let answer = data['message']
                    let sentiment = data['sentiment']
                    console.log(answer);
                    $('#sentiment-answer').html(answer);
                    if (sentiment == "Negative") {
                        $('#sentiment-answer').removeClass('text-success').addClass('text-danger')
                    } else {
                        $('#sentiment-answer').removeClass('text-danger').addClass('text-success')
                    }
                } else {
                    alert('Was not a success')
                }
            }
        })
    }
})

$('#diamond-submit').click(() => {
    let carat = $('#diamond-carat').val();
    let cut = $('#diamond-cut').val();
    console.log('Query is carat: ' + carat + ' cut: ' + cut);
    if (carat === null) {
        alert('Must input carat value to get prediction.');
    } else {
        let url = 'http://127.0.0.1:5000/api/diamond?carat=' + carat + '&cut=' + cut;
        console.log('Querying url: ' + url);
        $.ajax({
            url: url,
            success: function(data) {
                console.log(data);
                if (data['success']) {
                    let price = data['message']
                    $('#diamond-price').html(price);
                }
            }
        })
    }
})

$("#cat-dog-submit").click(() => {
    var fd = new FormData();
    var files = $('#file')[0].files[0];
    fd.append('file',files);

    $.ajax({
        url: 'http://127.0.0.1:5000/api/cat-and-dog',
        type: 'post',
        data: fd,
        contentType: false,
        processData: false,
        success: function(data){
            console.log(data)
            if (data['success']) {
                $("#cat-dog-answer").html(data['message'])
            } else {
                alert(data['message'])
            }
        },
    });
});